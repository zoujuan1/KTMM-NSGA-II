classdef ProbabilityModel % 
    properties
        modeltype; % multivariate normal ('mvarnorm' - for real coded) or ('umd' - for binary coded)
        mean_noisy;
        mean_true;
        covarmat_noisy;
        covarmat_true;
        probofone_noisy;
        probofone_true;
        probofzero_noisy;
        probofzero_true;
        vars;
    end
    methods (Static)
        function model = ProbabilityModel(type)
            model.modeltype = type;
        end
        function solutions = sample(model,nos)
            if strcmp(model.modeltype,'mvarnorm')
                solutions = mvnrnd(model.mean_noisy,model.covarmat_noisy,nos); %产生nos个多维正态随机数，前两个为期望向量，和协方差矩阵
            elseif strcmp(model.modeltype,'umd')
                solutions = rand(nos,model.vars);
                for i = 1:nos
                    index1 = solutions(i,:) <= model.probofone_true;
                    index0 = solutions(i,:) > model.probofone_true;
                    solutions(i,index1) = 1;
                    solutions(i,index0) = 0;
                end
            end
        end
        function solutions = sample_center(model,nos)
            if strcmp(model.modeltype,'mvarnorm')
                solutions = mvnrnd(model.mean_true,model.covarmat_true,nos); %产生nos个多维正态随机数，前两个为期望向量，和协方差矩阵
            end
        end
        function probofsols = pdfeval(model,solutions)
            if strcmp(model.modeltype,'mvarnorm')
                probofsols = mvnpdf(solutions,model.mean_noisy,model.covarmat_noisy);%如果solutions 中有110个解，则mvnpdf 返回这110个个体（110行）每个的密度函数值
            elseif strcmp(model.modeltype,'umd')
                nos = size(solutions,1);
                probofsols = zeros(nos,1);
                probvector = zeros(1,model.vars);
                for i = 1:nos
                    index = solutions(i,:) == 1;
                    probvector(index) = model.probofone_noisy(index);
                    index = solutions(i,:) == 0;
                    probvector(index) = model.probofzero_noisy(index);
                    probofsols(i) = prod(probvector);
                end
            end
        end
        function model = buildmodel(model,solutions)
            [pop,model.vars] = size(solutions);% 返回种群大小 pop=100，以及X变量维度30， 相当于返回行数和列数
            if strcmp(model.modeltype,'mvarnorm')
                model.mean_true = mean(solutions);% 100个X在每个X列上面的平均值
                covariance = cov(solutions);
                % dia(covariance)留下对角线的协方差，比如 对角线的第一个cov(x1,x1),
                % 第二个cov(x2,x2)  相当于求每个X分量的方差
                model.covarmat_true = diag(diag(covariance)); %
                % round 向下取整函数
                solutions_noisy = [solutions;rand(round(0.2*pop),model.vars)];%在solutions 后面加10行噪声，当pop = 100的时候
                model.mean_noisy = mean(solutions_noisy);
                covariance = cov(solutions_noisy);
                model.covarmat_noisy = diag(diag(covariance));% Simplifying to univariate distribution by ignoring off diagonal terms of covariance matrix
                %model.covarmat_noisy = cov(solutions_noisy);
            elseif strcmp(model.modeltype,'umd')
                model.probofone_true = mean(solutions);
                model.probofzero_true = 1 - model.probofone_true;
                solutions_noisy = [solutions;round(rand(round(0.1*pop),model.vars))];
                model.probofone_noisy = mean(solutions_noisy);
                model.probofzero_noisy = 1 - model.probofone_noisy;
            end
        end
    end
end