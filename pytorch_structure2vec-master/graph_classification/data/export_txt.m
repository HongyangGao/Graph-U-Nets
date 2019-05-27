clear

datafile = 'NCI1';

load(sprintf('%s/%s.mat', datafile, datafile));
label = label - min(label);
fid = fopen(sprintf('%s/%s.txt', datafile, datafile), 'w');
fprintf(fid, '%d\n', length(label));

for i = 1 : length(label)
    g = graph_struct(i);
    num_nodes = length(g.al);
    fprintf(fid, '%d %d\n', num_nodes, label(i));
    for j = 1 : num_nodes
        num_neighbors = length(g.al{j}); 
        fprintf(fid, '%d %d', g.nl.values(j) - 1, num_neighbors);
        for k = 1 : num_neighbors
            fprintf(fid, ' %d', g.al{j}(k) - 1);
        end
        fprintf(fid, '\n');
    end
end

fclose(fid);

total = length(label);
fold_size = floor(total / 10);
p = randperm(total);
for fold = 1 : 10
    test_range = (fold - 1) * fold_size + 1 : fold * fold_size;
    train_range = [1 : (fold - 1) * fold_size, fold * fold_size + 1 : total];
    
    fid = fopen(sprintf('%s/10fold_idx/test_idx-%d.txt', datafile, fold), 'w');
    for i = 1 : length(test_range)
        fprintf(fid, '%d\n', p(test_range(i)) - 1);
    end
    fclose(fid);

    fid = fopen(sprintf('%s/10fold_idx/train_idx-%d.txt', datafile, fold), 'w');
    for i = 1 : length(train_range)
        fprintf(fid, '%d\n', p(train_range(i)) - 1);
    end
    fclose(fid);
end
