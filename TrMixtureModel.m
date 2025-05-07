classdef TrMixtureModel % Works reliably for 2(+) Dimensional distributions
    properties
        model_list; % cell array of ProbabilityModels
        alpha; % weights of the models in stacking of mixture models
        noms; % number of models
        probtable; % Probability table required for stacking EM algorithm
        nsols; % number of solutions in probability table
    end
    methods (Static) % 初始化变量
        function mmodel = TrMixtureModel(allmodels)
            mmodel.model_list = allmodels;
            mmodel.noms = length(allmodels);
            mmodel.alpha = (1/mmodel.noms)*ones(1,mmodel.noms);% 初始化每个子模型的权重（被选择的概率），初始都是一样的
        end
        function mmodel = EMstacking(mmodel) % 最大期望算法EM，求系数大小
            iterations = 100;
            for i = 1:iterations
                talpha = mmodel.alpha;
                probvector = mmodel.probtable*talpha';
                for j = 1:mmodel.noms
                    talpha(j) = sum((1/mmodel.nsols)*talpha(j)*mmodel.probtable(:,j)./probvector);
                end
                mmodel.alpha = talpha;
            end
        end
        function mmodel = mutate(mmodel)
            modifalpha = max(mmodel.alpha+normrnd(0,0.01,[1,mmodel.noms]),0); %%%%%%%% Determining std dev for mutation can be a parameteric study %%%%%%%%%%%%%%%%
            pusum = sum(modifalpha);
            if pusum == 0 % Then complete weightage assigned to target model alone
                mmodel.alpha = zeros(1,mmodel.noms);
                mmodel.alpha(mmodel.noms) = 2;
            else
                mmodel.alpha = modifalpha/pusum;
            end
        end
        function mmodel = Similarity(mmodel, average_objvalue,K)
            mmodel.alpha = zeros(1,mmodel.noms);
            corr_value = zeros(1,mmodel.noms);
            for i = 1:(mmodel.noms-1)
                corr_value(i) = corr(average_objvalue(i,:)',average_objvalue(mmodel.noms,:)');
                if corr_value(i) < 0 || isnan(corr_value(i))
                    corr_value(i) = 0;
                end
            end
           if size(corr_value) <= K + 1
                corr_value(mmodel.noms) = 0.5;
                for j = 1:mmodel.noms
                    mmodel.alpha(j) = corr_value(j)/sum(corr_value);
                end
           else
                [~, sorted_indices] = sort(corr_value, 'descend');
                % 找出前K个最大值的索引
                top_indices = sorted_indices(1:K);
                % 将除了前K个最大值以外的值赋为0
                top_Obj = zeros(size(corr_value));
                top_Obj(top_indices) = corr_value(top_indices);
                top_Obj(mmodel.noms) = 0.5;
                for j = 1:mmodel.noms
                    mmodel.alpha(j) = top_Obj(j)/sum(top_Obj);
                end
            end
        end
        function mmodel = M1_Similarity(mmodel,K)
            mmodel.alpha = zeros(1,mmodel.noms);
            M1_similarity = zeros(1,mmodel.noms);
            % 计算source task 跟 target task 的相似，通过M1
            for i = 1:(mmodel.noms-1)
                source_mean = mmodel.model_list{1,i}.mean_noisy;
               target_mean = mmodel.model_list{1,mmodel.noms}.mean_noisy;
                M1_similarity(i) = 1/norm(source_mean-target_mean);
            end
            if size(M1_similarity) <= K + 1
                 M1_similarity =  M1_similarity / sum(M1_similarity);% 对M1进行归一化
                 M1_similarity(mmodel.noms) = 0.5;% target task 跟 target task的相似度参数
                for j = 1:mmodel.noms
                    mmodel.alpha(j) = M1_similarity(j)/sum(M1_similarity);
                end
            else
                 M1_similarity =  M1_similarity / sum(M1_similarity);% 对M1进行归一化
                [~, sorted_indices] = sort(M1_similarity, 'descend');
                  % 找出前K个最大值的索引
                top_indices = sorted_indices(1:K);
                % 将除了前K个最大值以外的值赋为0
                top_M1 = zeros(size(M1_similarity));
                top_M1(top_indices) = M1_similarity(top_indices);
                top_M1(mmodel.noms) = 0.5;
                for j = 1:mmodel.noms
                    mmodel.alpha(j) = top_M1(j)/sum(top_M1);
                end
            end
         end
         function mmodel = KLD_Similarity(mmodel,K)
            mmodel.alpha = zeros(1,mmodel.noms);
            KLD_similarity = zeros(1,mmodel.noms);
            % 计算source task 跟 target task 的相似，通过KLD
            for i = 1:(mmodel.noms-1)
                 source_mean = mmodel.model_list{1,i}.mean_noisy;
                 source_cov = mmodel.model_list{1,i}.covarmat_noisy;
                 target_mean = mmodel.model_list{1,mmodel.noms}.mean_noisy;
                 target_cov = mmodel.model_list{1,mmodel.noms}.covarmat_noisy;
                 KLD = 0.5*(trace(inv(source_cov)*target_cov)+(source_mean-target_mean)*...
                source_cov*(source_mean-target_mean)'...
                -length(source_mean)+log(det(source_cov)/det(target_cov)));
                KLD_similarity(i) = 1/KLD;
            end
            if size(KLD_similarity) <= K + 1
                 KLD_similarity =  KLD_similarity / sum(KLD_similarity);% 对KLD进行归一化
                 KLD_similarity(mmodel.noms) = 0.5;% target task 跟 target task的相似度参数
                for j = 1:mmodel.noms
                    mmodel.alpha(j) = KLD_similarity(j)/sum(KLD_similarity);
                end
            else
                %lastX_Values = WD_similarity(end-(last_number+1):end);
                KLD_similarity =  KLD_similarity / sum(KLD_similarity);% 对KLD进行归一化
                [~, sorted_indices] = sort(KLD_similarity, 'descend');
                  % 找出前K个最大值的索引
                top_indices = sorted_indices(1:K);
                % 将除了前K个最大值以外的值赋为0
                top_KLD = zeros(size(KLD_similarity));
                top_KLD(top_indices) = KLD_similarity(top_indices);
                top_KLD(mmodel.noms) = 0.5;
                for j = 1:mmodel.noms
                    mmodel.alpha(j) = top_KLD(j)/sum(top_KLD);
                end
            end
        end
        function mmodel = WD_Similarity(mmodel,K)
            mmodel.alpha = zeros(1,mmodel.noms);
            WD_similarity = zeros(1,mmodel.noms);
            % 计算source task 跟 target task 的相似，通过WD
            for i = 1:(mmodel.noms-1)
                mean_distance = norm(mmodel.model_list{1,i}.mean_noisy - mmodel.model_list{1,mmodel.noms}.mean_noisy);
                vari_distance = norm(mmodel.model_list{1,i}.covarmat_noisy - mmodel.model_list{1,mmodel.noms}.covarmat_noisy);
                WD_similarity(i) = 1/sqrt(mean_distance + vari_distance);
            end
            if size(WD_similarity) <= K+1 
                WD_similarity(mmodel.noms) = 0.5;
                for j = 1:mmodel.noms
                    mmodel.alpha(j) = WD_similarity(j)/sum(WD_similarity);
                end
            else
                WD_similarity =  WD_similarity / sum(WD_similarity);% 对WD进行归一化
                [~, sorted_indices] = sort(WD_similarity, 'descend');
                  % 找出前K个最大值的索引
                top_indices = sorted_indices(1:K);
                
                % 将除了前K个最大值以外的值赋为0
                top_WD = zeros(size(WD_similarity));
                top_WD(top_indices) = WD_similarity(top_indices);
                top_WD(mmodel.noms) = 0.5;
                for j = 1:mmodel.noms
                    mmodel.alpha(j) = top_WD(j)/sum(top_WD);
                end
            end
        end
        function solutions = sample(mmodel,nos)
            indsamples = ceil(nos*mmodel.alpha);
            totalsamples = sum(indsamples);
            solutions = [];
            for i = 1:mmodel.noms
                if indsamples(i) == 0
                    continue;
                else
                    sols = ProbabilityModel.sample(mmodel.model_list{i},indsamples(i));
                    solutions = [solutions; sols];
                end
            end
            solutions = solutions(randperm(totalsamples),:);
            solutions = solutions(1:nos,:);
        end
        function solutions = sample_center(mmodel,nos)
            indsamples = ceil(nos*mmodel.alpha);
            totalsamples = sum(indsamples);
            solutions = [];
            for i = 1:mmodel.noms
                if indsamples(i) == 0
                    continue;
                else
                    sols = ProbabilityModel.sample_center(mmodel.model_list{i},indsamples(i));
                    solutions = [solutions; sols];
                end
            end
            solutions = solutions(randperm(totalsamples),:);
            solutions = solutions(1:nos,:);
        end
        function mmodel = createtable(mmodel,solutions,CV,type)
            if CV
                mmodel.noms = mmodel.noms+1; %%%%%% NOTE: Last model in the list is the target model
                mmodel.model_list{mmodel.noms} = ProbabilityModel(type);
                mmodel.model_list{mmodel.noms} = ProbabilityModel.buildmodel(mmodel.model_list{mmodel.noms},solutions);% 这里建立了target task 的独立高斯模型，根据solutions,至于solutions怎么来有一些不同的方式。
                mmodel.alpha = (1/mmodel.noms)*ones(1,mmodel.noms);
                nos = size(solutions,1);% nos = 110
                mmodel.probtable = ones(nos,mmodel.noms); % noms行 * mmodel.noms列
                for j =1:mmodel.noms-1
                    mmodel.probtable(:,j) = ProbabilityModel.pdfeval(mmodel.model_list{j},solutions);% 计算当前个体集在历史环境建立的高斯模型中的密度函数值
                end
                for i = 1:nos % Leave-one-out cross validation scheme
                    x = [solutions(1:i-1,:);solutions(i+1:nos,:)];% 去掉第i个个体
                    tmodel = ProbabilityModel(type);
                    tmodel = ProbabilityModel.buildmodel(tmodel,x);% 去掉第i个体，剩下的作为训练集建立多元高斯分布模型，
                    mmodel.probtable(i,mmodel.noms) = ProbabilityModel.pdfeval(tmodel,solutions(i,:)); % 计算第i个个体的密度函数值
                end
            else
                nos = size(solutions,1);
                mmodel.probtable = ones(nos,mmodel.noms);
                for j =1:mmodel.noms
                    mmodel.probtable(:,j) = ProbabilityModel.pdfeval(mmodel.model_list{j},solutions);
                end
            end
            mmodel.nsols = nos;
        end
    end
end