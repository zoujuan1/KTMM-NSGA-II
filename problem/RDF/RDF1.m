classdef RDF1 < PROBLEM
    % <multi> <real> <large/none> <dynamic>
    % Benchmark dynamic MOP proposed by Shouyong Jiang1, Shengxiang Yang,et
    % taut --- 10 --- Number of generations for static optimization
    % nt   --- 10 --- Number of distinct steps
    
    %------------------------------- Reference --------------------------------
    % M. Farina, K. Deb, and P. Amato, Dynamic multiobjective optimization
    % problems: Test cases, approximations, and applications, IEEE Transactions
    % on Evolutionary Computation, 2004, 8(5): 425-442.
    %------------------------------- Copyright --------------------------------
    % Copyright (c) 2022 BIMK Group. You are free to use the PlatEMO for
    % research purposes. All publications which use this platform or any code
    % in the platform should acknowledge the use of "PlatEMO" and reference "Ye
    % Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
    % for evolutionary multi-objective optimization [educational forum], IEEE
    % Computational Intelligence Magazine, 2017, 12(4): 73-87".
    %--------------------------------------------------------------------------
    
    properties
        taut;       % Number of generations for static optimization
        nt;         % Number of distinct steps
        Optimums;   % Point sets on all Pareto fronts
        lower_limit; % the lower limit of random sequence
        upper_limit; % the upper limit of random sequence
        random_integers; % random sequence
    end
    methods
        %% Default settings of the problem
        function Setting(obj)
            [obj.taut,obj.nt] = obj.ParameterSet(10,10);
            obj.M = 2;
            if isempty(obj.D); obj.D = 10; end
            obj.lower    = zeros(1,obj.D);
            obj.upper    = ones(1,obj.D);
            obj.encoding = ones(1,obj.D);
            rng(42);
            obj.lower_limit = 0;
            obj.upper_limit = obj.maxFE/obj.N/obj.taut + 1; % number of environments = number of changes + 1
            obj.random_integers = randperm(obj.upper_limit - obj.lower_limit+1, obj.upper_limit) + obj.lower_limit - 1;
        end
        %% Evaluate solutions
        function Population = Evaluation(obj,varargin)
            PopDec     = obj.CalDec(varargin{1});
            PopObj     = obj.CalObj(PopDec);
            PopCon     = obj.CalCon(PopDec);
            % Attach the current number of function evaluations to solutions
            Population = SOLUTION(PopDec,PopObj,PopCon,zeros(size(PopDec,1),1)+obj.FE);
            obj.FE     = obj.FE + length(Population);
          
        end

        %% Calculate objective values
        function PopObj = CalObj(obj,PopDec)
            PopObj(:,1) = PopDec(:,1);
           % t = floor(obj.FE/obj.N/obj.taut)/obj.nt;
            index_t = mod(floor(obj.FE/obj.N/obj.taut), obj.upper_limit);
            Q_t = obj.random_integers(:,index_t + 1);
            t = Q_t/obj.nt;
            G = abs(sin(0.5*pi*t));
            H = 0.75*sin(0.5*pi*t)+1.25;
            g = 1 + sum((PopDec(:,2:end) - G).^2, 2);
            h = 1-(PopObj(:,1)./g).^H;
            PopObj(:,2) = g.*h;
        end
        %% Generate points on the Pareto front
        function R = GetOptimum(obj,N)
           % t = floor(0:obj.maxFE/obj.N/obj.taut)/obj.nt;
            index_t = mod(floor(0:obj.maxFE/obj.N/obj.taut), obj.upper_limit);
            Q_t = obj.random_integers(:,index_t + 1);
            t = Q_t/obj.nt;
            G = 0.75*sin(0.5*pi*t)+1.25;
            G = round(G*1e6)/1e6;
            x = linspace(0,1,N)';
            obj.Optimums = {};
            for i = 1 : length(G)
                g = G(i);
                obj.Optimums(i,:) = {G(i),[x,1 - x.^g]};
            end
            % Combine all point sets
            R = cat(1,obj.Optimums{:,2});
        end
        %% Generate the image of Pareto front
        function R = GetPF(obj)
            R = obj.GetOptimum(1500);
        end
          %% Calculate the metric value
        function score = CalMetric(obj,metName,Population)
            t  = floor(0:obj.maxFE/obj.N/obj.taut)/obj.nt;
            Scores = zeros(1,length(t));
            for i = 1:obj.N:length(Population)
                subPop = Population(i:i + obj.N - 1);
                k = ceil(i/obj.N);
                Scores(k) = feval(metName,subPop,obj.Optimums{k, 2});
            end
            Score_IGD=[log10(Scores)]; %存MIGD
            save("E:\动态偏好\Knowledge-Transfer-GMM\Code\PlatEMO-master\PlatEMO\IGD\DIPDF1"+".mat","Score_IGD");
            score = mean(Scores);
%             fileID = fopen('E:\动态偏好\Knowledge-Transfer-GMM\Code\PlatEMO-master\PlatEMO\20IGD-RDF\IGD_RDF1.txt', 'a'); % 'a' 表示以追加模式打开文件
%             fprintf(fileID, '%0.5f\n', score);
%             fclose(fileID);
        end
%         %% Calculate the metric value
%         function score = CalMetric(obj,metName,Population)
%             %t      = floor(Population.adds/obj.N/obj.taut)/obj.nt;
%             t      = floor(Population.adds/obj.N/obj.taut)/obj.nt;
%             G      = 0.75*sin(0.5*pi*t)+1.25;
%             G      = round(G*1e6)/1e6;
%             change = [0;find(G(1:end-1)~=G(2:end));length(G)];
%             Scores = zeros(1,length(change)-1);
%             allG   = cell2mat(obj.Optimums(:,1));
%             for i = 1 : length(change)-1
%                 subPop    = Population(change(i)+1:change(i+1));
%                 Scores(i) = feval(metName,subPop,obj.Optimums{find(G(change(i)+1)==allG,1),2});
%             end
%             score = mean(Scores);
%         end
    end
end