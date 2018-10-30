classdef Storage < handle
    properties 
        RNN_copy
        loss
    end
    methods
        function obj = Storage(RNN)
            obj.RNN_copy.b = zeros(size(RNN.b));
            obj.RNN_copy.W = zeros(size(RNN.W));
            obj.RNN_copy.V = zeros(size(RNN.V));
            obj.RNN_copy.c = zeros(size(RNN.c));
            obj.RNN_copy.U = zeros(size(RNN.U));
        end
        
        function initialize_loss(obj, smooth_loss)
            obj.loss = smooth_loss;
        end
        
        function StoreOnLowest(obj, smooth_loss, RNN)
            if smooth_loss < obj.loss
                obj.loss = smooth_loss;
                obj.RNN_copy = RNN;
            end
        end
        
        function WriteToDisc(obj)
            obj.RNN_copy;
            save('results/RNN_copy', 'obj.RNN_copy' ,'-mat');
        end
        
        function RNN = LoadFromDisc(obj)
            obj.RNN_copy = load('results/RNN_copy', '-mat');
            RNN = obj.RNN_copy;
        end
    end
end