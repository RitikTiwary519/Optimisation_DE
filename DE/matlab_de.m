clc;
clear;
close all;

%% Problem Definition

costfunction=@(x) Sphere(x);

dim=2;  % No. of decision variables 
down=-5; % Lower Bound        
up= 5;   % Upper Bound
VarSize=[1 dim];   % Decision Variables Matrix Size

%% DE Parameters

MaxIt=100;      % Maximum Number of Iterations

nPop=5;        % Population Size

F=0.5;   %Scaling Factor

pCR=0.7;        % Crossover Probability

%% Initialization
        
empty_individual.Position=[];
empty_individual.Cost=[];

BestSol.Cost=inf;

pop=repmat(empty_individual,nPop,1);

for i=1:nPop
  
    pop(i).Position=unifrnd(down,up,VarSize);
    
    pop(i).Cost=costfunction(pop(i).Position);
     
    if pop(i).Cost<BestSol.Cost
        BestSol=pop(i);
    end
  
end

BestCost=zeros(MaxIt,1);

%% DE Main Loop

for it=1:MaxIt
    
    for i=1:nPop
        
        x=pop(i).Position; % Parent
        
        A=randperm(nPop);  % Random Permutation of Number of variables.
        
       A(A==i)=[];   % A - parent number. 
                        %This is required to choose the target and other two random vectors
        
        a=A(1); % target vector
        b=A(2); % randomly selected vector
        c=A(3); % randomly selected vector
        
        
        % Mutation  :   u = trial vector; 

        u=pop(a).Position+F.*(pop(b).Position-pop(c).Position); % Trial Vector
        
        u = max(u, down); % keep the trial vector in the range
		u = min(u,up);
		
        
        % Crossover
        
        z=zeros(size(x));
        
        j0=randi([1 numel(x)]);  % Randomly selected integer from 1 to Dim
                                %for binomial crossover  
       
        % Binomial Crossover
        for j=1:numel(x)
            if j==j0 || rand<=pCR  % If j is either j0 or probability of 
                                    %crossover is true then 
                                    %corresponding varible will be taken
                                    %from the trial vector otherwise from
                                    %parent
                z(j)=u(j);
            else
                z(j)=x(j);
            end
        end
        
        NewSol.Position=z;  % Solution obtained after mutation and crossover
        NewSol.Cost=costfunction(NewSol.Position); % Corresponding function value
        
        
       % Greedy Selection: If the solution after applying mutation and
       % crossover is better then accept the same otherwise carry the ith
       % solution. This is required to make sure that the overall quality
       % of the population is not deteriorats. 
      
       if NewSol.Cost<pop(i).Cost 
            pop(i)=NewSol;
            
            if pop(i).Cost<BestSol.Cost
               BestSol=pop(i);
            end
       end
    
    end
    
    % Update Best Cost
    BestCost(it)=BestSol.Cost;
    
    % Show Iteration Information
   
    disp([ 'iter' num2str(it)  ' : BestSol.Cost ' num2str(BestSol.Cost)]);

end

 
 function z=Sphere(x)

    z=sum(x.^2);

end