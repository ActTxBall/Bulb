import numpy as np

o_lyr = 4
h_lyr = 1
x_inp=14
# x_inp=len(x_len)
mutation_50bps = True


def sigmoid_activation(x):
    return 1/(1+np.exp(-x))
def relu_activation(x):
    return  np.maximum(x,0)
def soft_max(x):
    return np.exp(x)/np.sum(np.exp(x))
def scale_input(x):
    # return (x-np.min(x))/(np.max(x)-np.min(x))
    return (x/1000)
class Nnet1:
    h_output = h_lyr
  
    
    def forward(self,x_inp,hidden_neurons = h_lyr):
        weights=[]
        bias=[]

        hidden_layer_neurons = hidden_neurons
        output_layer_neurons = o_lyr        

        input_layer = x_inp
        input_layer = np.array(input_layer).reshape(-1,1)
        input_layer = scale_input(input_layer)

        input_len = len(input_layer)
        weights.append(np.random.rand(hidden_layer_neurons,input_len)-.5)
        weights.append(np.random.rand(output_layer_neurons,hidden_layer_neurons)-.5)
        bias.append(np.random.rand(hidden_layer_neurons,1)-.5)    
        bias.append(np.random.rand(output_layer_neurons,1)-.5)
        hidden_layer = np.matmul(weights[0],input_layer) + bias[0]        

        hidden_layer = sigmoid_activation(hidden_layer)
        output_layer = np.matmul(weights[1],hidden_layer) + bias[1]
        output_layer = soft_max(output_layer)  

        return output_layer,weights,bias
    def lockin(self,x_inp,weights,bias):        

        input_layer = x_inp
        input_layer = np.array(input_layer).reshape(-1,1)
        input_layer = scale_input(input_layer)
        input_len = len(input_layer)      
        hidden_layer = np.matmul(weights[0],input_layer) + bias[0]
        hidden_layer = relu_activation(hidden_layer)
        output_layer = np.matmul(weights[1],hidden_layer) + bias[1]
        output_layer = soft_max(output_layer)        

        return output_layer
    def mutate(self,weights_bot_card,bias_bot_card,wt_pct,bi_pct,mut_amt = 1):
        wi_c =[]
        bi_c=[]
        for wi in weights_bot_card:
            xxw = np.round(np.random.rand(wi.shape[0],wi.shape[1])-.5,2)*mut_amt
            xx=np.where(np.random.rand(wi.shape[0],wi.shape[1])<wt_pct,xxw,0)
            wi_c.append(wi+xx)
        for bi in bias_bot_card:
            xxb=  np.round(np.random.rand(bi.shape[0],bi.shape[1])-.5,2)*mut_amt
            xxy=np.where(np.random.rand(bi.shape[0],bi.shape[1])<bi_pct,xxb,0)
            bi_c.append(bi+xxy)
        return [0,0],wi_c,bi_c
    def Crossover(self,parent1_weights,parent2_weights,parent1_bias,parent2_bias,type='one_point',hidden_neurons = h_lyr):
        
        child_gene_weight=[]
        child_gene_bias=[]
        cross_over_pct_action = .8
        cross_over_method_at_every_point = False
        one_pt_xover = False
        undefined_point_crossover = False
        crossover_points = 2
        crossover_points_bias = [min(3, hidden_neurons - 2), min(3, o_lyr - 2)]

        if type== 'all_point':
            cross_over_method_at_every_point = True
        elif type=='one_point':
            one_pt_xover = True
        elif type=='three_point':
            undefined_point_crossover = True

        if np.random.rand()<cross_over_pct_action:
            for wi1,wi2 in zip(parent1_weights, parent2_weights):
                if cross_over_method_at_every_point:
                    wi1 = wi1.reshape(-1)
                    wi2 = wi1.reshape(-1)
                    tmp_gene = []
                    for i,j in zip(wi1,wi2):
                        if np.random.rand()>=.5:
                            tmp_gene = np.append(tmp_gene,i)
                        else:
                            tmp_gene = np.append(tmp_gene,j)

                    child_gene_weight.append(tmp_gene)
                if undefined_point_crossover :
                    wi1= wi1.reshape(-1)
                    wi2 = wi1.reshape(-1)
                    # print(wi1)
                    splitloc = range(2, len(wi1)-1)
                    # print(splitloc)
                    splitloc = np.random.choice(splitloc,size=(crossover_points-1,),replace = False)
                    # print(splitloc)
                    splitloc = np.sort(splitloc)
                    # print(splitloc)
                    cut_gene1 = np.split(wi1,splitloc)
                    # print(cut_gene1)
                    cut_gene2 = np.split(wi2,splitloc)
                    # print('gene2:',cut_gene2)
                    tmp_gene=[]
                    for i,j in zip(cut_gene1,cut_gene2):
                        if np.random.rand()>.5:
                        # if True:
                            # prnt= i
                            # print(tmp_gene)
                            # print(i)
                            tmp_gene = np.append(tmp_gene,i)
                            # tmp_gene += i
                        else:
                            # prnt = j
                            tmp_gene = np.append(tmp_gene,j)

                    # print(child_gene_weight)
                    child_gene_weight.append(tmp_gene)
                    # print(child_gene_weight)
                if one_pt_xover :
                    start_cross = np.random.random_integers(0, len(wi1))
                    if np.random.rand()>.5:

                        x=wi1[0:start_cross]
                        y=wi2[start_cross:]
                        x=np.append(x,y)
                        child_gene_weight.append(x)
                    else:
                        x=wi2[0:start_cross]
                        y=wi1[start_cross:]
                        x=np.append(x,y)
                        child_gene_weight.append(x)
                #
            child_gene_weight[0] = np.array(child_gene_weight[0]).reshape(hidden_neurons ,x_inp)
            child_gene_weight[1] = np.array(child_gene_weight[1]).reshape(o_lyr,hidden_neurons )
            
            # bias crossover
            for wi1,wi2,cross_over_pt_bias in zip(parent1_bias, parent2_bias,crossover_points_bias):

                wi1 = wi1.reshape(-1)
                wi2 = wi1.reshape(-1)
                # one_pt_xover = type
                if one_pt_xover:
                    start_cross = np.random.random_integers(0, len(wi1))
                    if np.random.rand() > .5:

                        x = wi1[0:start_cross]
                        y = wi2[start_cross:]
                        x = np.append(x, y)
                        child_gene_bias.append(x)
                    else:
                        x = wi2[0:start_cross]
                        y = wi1[start_cross:]
                        x = np.append(x, y)
                        child_gene_bias.append(x)
                else:
                    splitloc = range(2, len(wi1) - 1)
                    splitloc = np.random.choice(splitloc, size=(cross_over_pt_bias - 1,), replace=False)
                    splitloc = np.sort(splitloc)
                    cut_gene1 = np.split(wi1, splitloc)
                    cut_gene2 = np.split(wi2, splitloc)
                    tmp_gene = []
                    for i, j in zip(cut_gene1, cut_gene2):
                        if np.random.rand() > .5:
                            tmp_gene = np.append(tmp_gene, i)
                        else:
                            tmp_gene = np.append(tmp_gene,j)
                    child_gene_bias.append(tmp_gene)
            child_gene_bias[0] = np.array(child_gene_bias[0]).reshape(hidden_neurons ,-1)
            child_gene_bias[1] = np.array(child_gene_bias[1]).reshape(o_lyr,-1)
            
            return [0,0],child_gene_weight,child_gene_bias
        else:
            return [0,0],parent1_weights,parent1_bias
