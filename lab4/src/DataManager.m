classdef DataManager
    properties
        book_data
        book_chars
        char_to_index = containers.Map('KeyType', 'char', 'ValueType', 'int32')
        index_to_char = containers.Map('KeyType', 'int32', 'ValueType', 'char')
        K 
    end
    methods
        function obj = DataManager(filename)
            book_fname = strcat('data/', filename);
            fid = fopen(book_fname, 'r');
            obj.book_data = fscanf(fid, '%c');
            fclose(fid);
            
            obj.book_chars = unique(obj.book_data);
            obj.K = length(obj.book_chars);
            
            for i = 1:obj.K
                obj.char_to_index(obj.book_chars(i)) = i;
                obj.index_to_char(i) = obj.book_chars(i);
            end
        end
        
        function [X_chars, Y_chars, X, Y] = one_hot_batch(obj, index, seq_length)
            start = index;
            stop = start + seq_length;
            X_chars = obj.book_data(start:stop);
            Y_chars = obj.book_data(start+1:stop+1);
            
            X = zeros(obj.K, stop - start);
            Y = zeros(obj.K, stop - start);
            
            for i = 1:seq_length
                X(obj.char_to_index(X_chars(i)),i) = 1;
                Y(obj.char_to_index(Y_chars(i)),i) = 1;
            end
        end
        
        function PrintText(obj, text)
            for i = 1:size(text,2)
                [~, argmax] = max(text(:,i));

                string(i) = obj.index_to_char(int32(argmax)); %#ok<AGROW>
            end
            disp(string)
        end
        
        function WriteText(obj, text, file)
            for i = 1:size(text,2)
                [~, argmax] = max(text(:,i));

                string(i) = obj.index_to_char(int32(argmax)); %#ok<AGROW>
            end
            fprintf(file, string, '\n\n');
        end
    end
end
