classdef RDF2 < PROBLEM
    % <multi> <real> <large/none> <dynamic>
    % Benchmark dynamic MOP proposed by Farina, Deb, and Amato
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
            %t = floor(obj.FE/obj.N/obj.taut)/obj.nt;
            index_t = mod(floor(obj.FE/obj.N/obj.taut), obj.upper_limit);
            Q_t = obj.random_integers(:,index_t + 1);
            t = Q_t/obj.nt;
            G = abs(sin(0.5*pi*t));
            r = 1 + floor((obj.D - 1)*G);
            PopObj(:,1) = PopDec(:,r);
            temp = setdiff(1:obj.D,r);
            g = 1 + sum((PopDec(:,temp)-sin(0.5.*pi.*t)).^2,2);
            h = 1 - sqrt(PopObj(:,1)./g);
            PopObj(:,2) = g.*h;
        end
        %% Generate points on the Pareto front
        function R = GetOptimum(obj,N)
            R(:,1) = linspace(0,1,N)';
            R(:,2) = 1 - R(:,1).^0.5;
        end
        %% Generate the image of Pareto front
        function R = GetPF(obj)
            R = obj.GetOptimum(100);
        end
    end
end