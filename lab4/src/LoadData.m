function [book_data, char_to_index, index_to_char, K] = LoadData(filename)
    book_fname = strcat('data/', filename);
    fid = fopen(book_fname, 'r');
    book_data = fscanf(fid, '%c');
    fclose(fid);
    
    book_chars = unique(book_data);
    K = length(book_chars);
    
    char_to_index = containers.Map(book_chars, 1:K);
    index_to_char = containers.Map(1:K, book_chars);
end