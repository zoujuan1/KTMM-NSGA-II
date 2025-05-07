classdef KTMMNSGAII < ALGORITHM
    % <multi> <real/integer/label/binary/permutation> <constrained/none> <dynamic>
    % KTMM + NSGAII for dynamic multi-objective problems
    
    %--------------------------------------------------------------------------
    % Copyright (c) 2016-2017 BIMK Group. You are free to use the PlatEMO for
    % research purposes. All publications which use this platform or any code
    % in the platform should acknowledge the use of "PlatEMO" and reference "Ye
    % Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
    % for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
    % Computational Intelligence Magazine, 2017, 12(4): 73-87".
    %--------------------------------------------------------------------------
    
    methods
        function main(Algorithm,Problem)
            %  全部用混合高斯模型产生
            % Reset the number of saved populations (only for dynamic optimization)
            % Algorithm.save = sign(Algorithm.save)*inf;
            allmodels = {}; % transfer stacking models
            %% Generate random population
            Population = Problem.Initialization();
            Problem.FE =  Problem.FE - length(Population);
            [Population,FrontNo,CrowdDis] = EnvironmentalSelection(Population,Problem.N);
            %T0环境先迭代同样次数，也可以不用，对比算法只要统一就行
%             for i = 1:Problem.N
%                 MatingPool = TournamentSelection(2,Problem.N,FrontNo,-CrowdDis);
%                 Offspring  = OperatorGA(Problem,Population(MatingPool));
%                 Problem.FE =  Problem.FE - length(Offspring);
%                 [Population,FrontNo,CrowdDis] = EnvironmentalSelection([Population,Offspring],Problem.N);
%             end
            %average_objvalue = zeros((Problem.maxFE/Problem.N/Problem.taut),Problem.M);
            AllPop = [];
            %AllPop1 = [];
            K = 14;% 利用历史环境高斯子模型的个数
           % Wtarget = 0.5;% Target task 在混合高斯模型的权重大小
           % AllPop1 = [AllPop1,Population];
            %% Optimization
            while Algorithm.NotTerminated(Population) % 
                if Changed(Problem,Population)
%                     average_objvalue(T,:) = mean(Population.objs,1); % 上次环境的最优解，保存上次环境目标空间中最优解的中心点
                    AllPop = [AllPop,Population];
                    
                    Population_last = EnvironmentalSelection(Population,Problem.N/2); % 非支配排序 环境选择出100个个体用于混合模型;
                   % Population_last = GoodSolutionSelection(Population,0.5*Problem.N);
                    % 对上次环境的最优解建立上次环境的模型
                    model = ProbabilityModel('mvarnorm');
                    model = ProbabilityModel.buildmodel(model, Population_last.decs);
                    allmodels{length(allmodels)+1} = model;
                    save('allmodels','allmodels');
                   % 对当前环境建立近似模型      
                    Population_new = Problem.RandomInitialization(Problem.N); 
                    Problem.FE =  Problem.FE - length(Population_new);
                    [Population,~,~] = EnvironmentalSelection(Population_new,Problem.N); % 非支配排序 环境选择出100个个体用于混合模型
                    %average_objvalue(T+1,:) = mean(Population.objs,1); % 上次环境的最优解，保存上次环境目标空间中最优解的中心
                    %Population  = GoodSolutionSelection(Population_new,Problem.N);
                    Pop_decs = Population.decs;
                    load('./allmodels')
                    mmodel = TrMixtureModel(allmodels);
                    mmodel = TrMixtureModel.createtable(mmodel, Pop_decs, true, 'mvarnorm');
                    %mmodel =  TrMixtureModel.EMstacking(mmodel);
                    %mmodel = TrMixtureModel.Similarity(mmodel, average_objvalue,K);
                    %mmodel = TrMixtureModel.KLD_Similarity(mmodel,K);
                   % mmodel = TrMixtureModel.M1_Similarity(mmodel,K);
                   %% knowledge fusion
                    mmodel = TrMixtureModel.WD_Similarity(mmodel, K);
                    %% Knowledge transfer             
                    offspring = TrMixtureModel.sample(mmodel,Problem.N);
                    offspring = Problem.Evaluation(offspring);
                    Problem.FE =  Problem.FE - length(offspring);
                    [Population,FrontNo,CrowdDis] = EnvironmentalSelection(offspring,Problem.N);
                   % Initial_IGD = IGD(Population.objs,Problem.Optimums{T,2});
                   % AllPop1 = [AllPop1,Population];
                    
                end
                MatingPool = TournamentSelection(2,Problem.N,FrontNo,-CrowdDis);
                Offspring  = OperatorGA(Problem,Population(MatingPool));
                [Population,FrontNo,CrowdDis] = EnvironmentalSelection([Population,Offspring],Problem.N);
                if  Problem.FE == (Problem.maxFE + Problem.N*Problem.taut)
                    % Return all populations
                    Population = [AllPop,Population];
                    %Population = AllPop1;
                    %                     [~,rank]   = sort(Population.adds(zeros(length(Population),1)));
                    %                     Population = Population(rank);
                end
            end
        end
    end
end










