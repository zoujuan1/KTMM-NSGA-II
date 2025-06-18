classdef RDF12 < PROBLEM
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
            obj.M = 3;
            if isempty(obj.D); obj.D = 10; end
            obj.lower    = [0,0,-ones(1,obj.D-2)];
            obj.upper    = [1,1,ones(1,obj.D-2)];
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
            index_t = mod(floor(obj.FE/obj.N/obj.taut), obj.upper_limit);
            Q_t = obj.random_integers(:,index_t + 1);
            t = Q_t/obj.nt;
            k = 10*sin(pi*t);
            k = round(k*1e6)/1e6;
            temp1 = sum( ( PopDec(:,3:end) - sin(t*PopDec(:,1)) ).^2, 2);
            temp2 = abs(sin(floor(k*(2*PopDec(:,1) - 1))*pi/2).*sin(floor(k*(2*PopDec(:,2) - 1))*pi/2));
            g = 1 + temp1 + temp2;
            PopObj(:,1) = g.*cos(0.5*pi*PopDec(:,2)).*cos(0.5*pi*PopDec(:,1));
            PopObj(:,2) = g.*sin(0.5*pi*PopDec(:,2)).*cos(0.5*pi*PopDec(:,1));
            PopObj(:,3) = g.*sin(0.5*pi*PopDec(:,1));
        end
        %% Generate points on the Pareto front
        function R = GetOptimum(obj,N)
            h = 50;
            index_t = mod(floor(0:obj.maxFE/obj.N/obj.taut), obj.upper_limit);
            Q_t = obj.random_integers(:,index_t + 1);
            t = Q_t/obj.nt;
            k = 10*sin(pi*t);
            k = round(k*1e6)/1e6;
            [X,Y] = meshgrid(linspace(0,1,h));
            PS = [];
            for i = 1:size(X,1)
                PS = [PS
                    [X(:,i) Y(:,i)]];
            end
            x1 = PS(:,1);
            x2 = PS(:,2);
            obj.Optimums = {};
            for i = 1 : length(k)
                temp = abs(sin(floor(k(i)*(2*x1-1))*pi/2).*sin(floor(k(i)*(2*x2-1))*pi/2));
                g = 1 + temp;
                f1 = g.*cos(0.5*pi*x2).*cos(0.5*pi*x1);
                f2 = g.*sin(0.5*pi*x2).*cos(0.5*pi*x1);
                f3 = g.*sin(0.5*pi*x1);
                R(:,1) = f1;
                R(:,2) = f2;
                R(:,3) = f3;
                R      = R(NDSort(R,1)==1,:);
                obj.Optimums(i,:) = {k(i),[ R ]};
                R = [];
                %obj.Optimums(i,:) = {G(i),[f1,f2,f3]};
            end
            % Combine all point sets
            R = cat(1,obj.Optimums{:,2});
        end
        %% Generate the image of Pareto front
        function R = GetPF(obj)
            R = obj.GetOptimum(100);
            %R(NDSort(R,1)>1,:) = nan;
        end
    end
end
