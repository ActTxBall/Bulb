
import pygame
import pickle
import numpy
from NNet import Nnet1

size=[50,50]
pygame.init()
coll_box=False
game_over = False
clock = pygame.time.Clock()

def mouse_in_box(box):
    if (box[0]<mouse[0]<(box[0]+box[2])) & (box[1]<mouse[1]<(box[1]+box[3])):
        return True
    return False
def draw_boxes():
    for boxy in boxOb:
        pygame.draw.rect(screen,boxy.color,(boxy.loc[0],boxy.loc[1],boxy.size,boxy.size))
def draw_players():
        for i in bot:
            # pygame.draw.rect(screen,(255,0,0),(bot[0].loc[0],bot[0].loc[1],50,50))
            pygame.draw.rect(screen,(255,0,0),(i[0].loc[0],i[0].loc[1],50,50))

def move_box(boxy):
        boxy.loc[0],boxy.loc[1] = (pygame.mouse.get_pos())
        boxy.loc[0],boxy.loc[1] = boxy.loc[0]-25,boxy.loc[1]-25
def collission_box(i):
    for x in boxOb:
        if ((x.loc[0]-50 < i[0].loc[0] < (x.loc[0]) ) and (x.loc[1] < i[0].loc[1] < (x.loc[1])+50 ))\
            or ((x.loc[0] < i[0].loc[0] < (x.loc[0]+50) ) and (x.loc[1] < i[0].loc[1] < (x.loc[1])+50 ))\
            or ((x.loc[0]-50 < i[0].loc[0] < (x.loc[0]) ) and (x.loc[1]-50 < i[0].loc[1] < (x.loc[1]) ))\
            or ((x.loc[0] < i[0].loc[0] < (x.loc[0]+50) ) and (x.loc[1]-50 < i[0].loc[1] < (x.loc[1]) )):
                return True
                
def collission_box_and_line_sight(i):

    coll_box=False
    ln_right = [1000]
    ln_up = [1000]
    ln_down = [1000]
    ln_left = [1000]

    for x in boxOb:

        # find if collission
        if ((x.loc[0]-50 < i[0].loc[0] < (x.loc[0]) ) and (x.loc[1] < i[0].loc[1] < (x.loc[1])+50 ))\
            or ((x.loc[0] < i[0].loc[0] < (x.loc[0]+50) ) and (x.loc[1] < i[0].loc[1] < (x.loc[1])+50 ))\
            or ((x.loc[0]-50 < i[0].loc[0] < (x.loc[0]) ) and (x.loc[1]-50 < i[0].loc[1] < (x.loc[1]) ))\
            or ((x.loc[0] < i[0].loc[0] < (x.loc[0]+50) ) and (x.loc[1]-50 < i[0].loc[1] < (x.loc[1]) )):
                coll_box = True
        # check box collission vision
        # if y-axis collision detected
        if ((x.loc[1]-25) < (i[0].loc[1]+25) < (x.loc[1]+75)):
            # if left side of box is to right of right side of bot
            if (x.loc[0]) > (i[0].loc[0]+50):
                ln_right.append((x.loc[0]) - (i[0].loc[0]+50))
            if (x.loc[0]+50) < (i[0].loc[0]):
                ln_left.append( (i[0].loc[0]) - (x.loc[0]+50) )
        # if x-axis collission
        if ((x.loc[0]-25) < (i[0].loc[0]+25) < (x.loc[0]+75)):
            # if top of box below bottom of bot
            if (x.loc[1]) > (i[0].loc[1]+50):
                ln_down.append((x.loc[1]) - (i[0].loc[1]+50))
            # if bottom of box above top of bot
            if (x.loc[1]+50) < (i[0].loc[1]):
                ln_up.append((i[0].loc[1])-(x.loc[1]+50))
                # print(abs(x.loc[1]+25) - i[0].loc[1])
        # print(ln)
        # print(min(ln))
        # ln.m        
        # y= ln_y[ln_y2.index(min(ln_y2))]
        y1=min(ln_down)
        y2=min(ln_up)
        # print(min(ln_y2))
        x1=min(ln_right)
        x2=min(ln_left)
        coll_bx =[x1,x2,y1,y2]
        danger = [ (i<21)*1000 for i in [x1,x2,y1,y2]]
    # print(ln_y,ln_y2)
    # return 0,0
    # print(xx,y)
    return x1,x2,y1,y2,coll_box,danger[0],danger[1],danger[2],danger[3]
def diag_line(i):
    ln_45=[6]
    ln_135=[6]
    ln_225=[6]
    ln_315=[6]
    bot_x=i[0].loc[0]
    bot_y = i[0].loc[1]
    for x in boxOb:
        # if y-axis collision detected
        for num in range(6):
            if ( bot_x+50+50*num < x.loc[0] < bot_x+100+50*num) & (bot_y-50 -50*num< x.loc[1] <bot_y-50*num):
                ln_135.append(num)
            if ( bot_x -50 -50*num < x.loc[0] < bot_x-50*num ) & (bot_y-50 -50*num< x.loc[1] <bot_y-50*num):
                # print(ln_45,num)
                ln_45.append(num)
                # print(ln_45)
            if ( bot_x+50-50*num < x.loc[0] < bot_x+100-50*num) & (bot_y+50 +50*num< x.loc[1] <bot_y+100+50*num):
                ln_225.append(num)
            if ( bot_x-50 -50*num< x.loc[0] < bot_x-50*num) & (bot_y+50+50*num < x.loc[1] <bot_y+100+50*num):
                ln_315.append(num)
    ln_45 = min(ln_45)
    ln_135 = min(ln_135)
    ln_225 = min(ln_225)
    ln_315 = min(ln_315)
    # print(ln_45,ln_135,ln_225,ln_315)    
    return ln_45,ln_135,ln_225,ln_315
def dist_nearest_box(i):
    array_dist_x.clear()
    array_dist_y.clear()
    array_dist_xy.clear()
    for x in boxOb:
        # x_d = x.loc[0] - (i[0].loc[0]+60)
        # y_d = i[0].loc[1] - (x.loc[1]+60)
        x_d = (x.loc[0]+25) - (i[0].loc[0]+25)
        y_d = (i[0].loc[1]+25) - (x.loc[1]+25)
        # y_d = x.loc[1] i[0].loc[1] - (
        array_dist_x.append(x_d) 
        array_dist_y.append(y_d) 
        array_dist_xy.append(abs(x_d)+abs(y_d)) 
        # print(array_dist)
    # print(min(array_dist_xy),array_dist_xy.index(min(array_dist_xy)),array_dist_x[105],array_dist_y[105])
    # array_dist_x[105],array_dist_y[105]
    pos=array_dist_xy.index(min(array_dist_xy))
    # asdas
    return array_dist_x[pos],array_dist_y[pos]
def predict_hit_box(i):
    bot_x =i[0].loc[0];bot_y = i[0].loc[1];
    left=False;right=False;up=False;down=False
    mvmt_size=60

    return left,right,up,down

def run_ai2(x_val,bot):
        # print(bot[2][1],bot[2][2])    
        output_net_p = bot[1].lockin(x_val,bot[2][1],bot[2][2])
        # if run_one_bot: print('{.2f}'.format(output_net_p));
        bot[6].append(numpy.round(output_net_p,2))
        if diagnostic_move:
            # print(numpy.round(output_net_p,2)),print('left','\n','up','\n','down','\n','right');
            total_move_probs.append(numpy.round(output_net_p,2))

        if all_or_nothing_output_neuron_fire:
            output_net = output_net_p.argmax()+1
            if(output_net == 1):
                bot[0].loc[0]+=-10
                # print('left')
            elif(output_net == 2):
                bot[0].loc[1]+=-10
                # print('up')
            elif(output_net == 3):
                bot[0].loc[1]+=+10
                # print('down')
            elif(output_net == 4):
                bot[0].loc[0]+=10
        else:
            # output_net_p
            # output_net = 
            # r =numpy.random.rand()
            xxy=output_net_p.reshape(-1)
            # print(xxy)
            r = numpy.random.choice([1,2,3,4],size=1,p=xxy)
            # print(r)
            if r ==1 :
                bot[0].loc[0]+=-10
            elif r ==2 :
                bot[0].loc[1]+=-10
            elif r ==3:
                bot[0].loc[1]+=+10
            elif r ==4:
                bot[0].loc[0]+=10
            
        # print(bot[1].lockin(x_val,bot[2][1],bot[2][2]))
        # print(output_net_p[3])
        
def load_game():
    # savename = "racetrack.pk"
    
    f = open(level_load, 'rb')
    up = pickle.Unpickler(f)
    global boxOb
    boxOb = up.load()
    f.close()
    # return boxOb

def track_data(x_val):
    global pos_data
    pos_data.append(x_val)

def save_best_genes_mutants(i):
    start_gene = i
    savename = "genes.pk"
    f = open(savename, 'wb')
    p = pickle.Pickler(f)
    p.dump(start_gene)
    f.close()

def save_all_parents_curr_gen(bots,savenamestr='genes_cross.pk'):
    start_gene_parents.clear()
    for i in bots:
        start_gene_parents.append(i)
    savename = savenamestr;f = open(savename, 'wb');p = pickle.Pickler(f);p.dump(start_gene_parents);f.close()


def quadrant(i):
    x=i[0].loc[0]
    y=i[0].loc[1]
    z=0
    if (y<=y_quad) & (x<=x_quad): z=1;
    elif (y<=y_quad) & (x>=x_quad): z=2;
    elif (y>=y_quad) & (x>=x_quad): z=3;
    elif (y>=y_quad) & (x<=x_quad): z=4;
    return z

    
def Generate_oneoffspring(parent1,parent2,mutate = True):

    hist_dec = []
    child_genes = Nnet1().Crossover(parent1[2][1],parent2[2][1],parent1[2][2],parent2[2][2],type=Cross_type,hidden_neurons = h_input)
    # Cross_type = 'one_point / three_point / all_point'
    if mutate: child_genes = Nnet1().mutate(child_genes[1],child_genes[2],mut_pct,mut_pct,mut_amt=mut_amt1);
    last_pos.clear()
    hist_dec.clear()
    if random_starts_y_axis :

        x = [Player([100,numpy.random.random_integers(499,501)]),Nnet1(),child_genes,you_lose,rew,last_pos,hist_dec]
    else:

        x = [Player([100, 500]), Nnet1(), child_genes, you_lose, rew, last_pos,hist_dec]
    return x

def Generate_nextgen():
    hist_dec=[]
    savename = "genes_cross.pk";
    f = open(savename, 'rb');
    up = pickle.Unpickler(f);
    start_gene_parents = up.load();
    f.close()
    # sav_gen.append((start_gene[2][1], start_gene[2][2]))
    randbot=[]
    randbot.clear()
    childbot =[]
    childbot.clear()
    parentbot = []
    parentbot.clear()
    keepbot =[]
    keepbot.clear()
    global bot
    bot.clear()
    last_pos.clear()
    parent_pool = start_gene_parents

    pct = int(.15 * len(parent_pool))
    if (gen== 3) or ((gen== 1) & (sim==0)):
        # print('inhere')
        for i in range(num_bots):
            y = [Player([100,500]),Nnet1(),Nnet1().forward(x_input_param,hidden_neurons = h_input),you_lose,rew,last_pos,hist_dec]
            randbot.append(y)
        # bot = randbot
    else:
        if num_bots == 1:
            parentbot.append(Generate_oneoffspring(parent1=parent_pool[0], parent2=parent_pool[0]))
            # bot = parentbot
        elif asexual_reproduce & False:
            pct_to_keep = .20
            # generate child with mutation
            for i in range(num_bots):
                yx = Generate_oneoffspring(parent1=parent_pool[0], parent2=parent_pool[0])
                childbot.append(yx)

            # keep some pct of bots
            num_keep_parent=int(pct_to_keep * len(parent_pool))
            for i in range(num_keep_parent):
                pp = Generate_oneoffspring(parent1=parent_pool[i], parent2=parent_pool[i],mutate=False)
                keepbot.append(pp)
            bot = keepbot
            # print(num)
            # print(len(bot))
            # bot = childbot
            num_left = num_bots - num_keep_parent
            # print(num_left)
            # add back in bots from child to make up remainder
            bot_c = childbot[0:num_left]
            bot.extend(bot_c)
            # print(len(bot))
        elif (gen==2) & (sim==0) :
            for i in range(num_bots):
                parentbot.append(Generate_oneoffspring(parent1=parent_pool[i], parent2=parent_pool[i]))
        elif True:
            probs_raw =[i[4] for i in parent_pool] # set prob based on fitness
            # probs_raw = (1+numpy.array(probs_raw))**4 # scale it up to power for 4
            probs_raw = (1 + numpy.array(probs_raw)) ** 100 # scale it up to power for 4
            probs_raw = numpy.array(probs_raw) # turn it into Numpy arraw
            probs = probs_raw/probs_raw.sum() #normalize to make probs = 1
            # print(numpy.array(probs).round(2))

            # print(round(probs,2))
            # print(len(probs))
            # print(probs_raw)
            # print(probs)
            # print(len(probs), len(probs1), num_pairs)
            # print(len(probs))
            for i in range(num_bots):
                Parentone_and_two = numpy.random.choice(range(0,num_bots), (2,), replace=True, p=probs)
                # print(Parentone_and_two)
                child_of2 = Generate_oneoffspring(parent1 = parent_pool[Parentone_and_two[0]] , parent2 =  parent_pool[Parentone_and_two[1]],mutate=True)
                parent_x = Generate_oneoffspring(parent1=parent_pool[i], parent2=parent_pool[i],mutate=False)
                # print(child_of2)
                # print(parent_x)
                childbot.append(child_of2)
                parentbot.append(parent_x)
                # print(childbot)
                # print(parentbot)
            # print(parentbot[0])
    return childbot,parentbot

def Recombination(childbot,parentbot):
    if ((gen==1) & (sim==0)) or  (gen==3) :
        pct_random = 1.00
        pct_unchanged = 0

    elif (gen==2) & (sim==0):
        pct_unchanged=1.0
        pct_random = 0.0
    else:
        # pct_random = 0.10
        # pct_unchanged = 0.8
        pct_random = 0.01
        pct_unchanged = 0.01


    pct_child = (1-pct_random-pct_unchanged)

    x=int(pct_unchanged*num_bots) # unchanged parents
    y=int(pct_child*num_bots) # child bots
    z=int(num_bots-x-y) # random bots
    # print(len(bot),x,y,z,sim,gen)
    bot.clear()
    hist_dec =[]
    # rew=[]
    you_lose=[]


    for i in range(z):
        bot.append([Player([100, 500]), Nnet1(), Nnet1().forward(x_input_param,hidden_neurons = h_input), you_lose, rew, last_pos, hist_dec])
    for i in range(x):
        bot.append(parentbot[i])
    for i in range(y):
        bot.append(childbot[i])
    # print(x)

class BoxPipeData:
    def __init__(self,loc,pIn,pOut,store,name,color,size):
        self.loc = loc
        self.pIn = pIn
        self.pOut = pOut
        self.store = store
        self.name = name
        self.color = color
        self.size = size

        
class Player:
    def __init__(self,loc,weights=0,bias=0,reward=0,move_hist=0) :
        self.loc = loc
        self.weights = weights
        self.bias = bias
        self.reward = reward
        self.move_hist = move_hist

    def calc_reward(self):
        pass

    def mutate_self(self):
        pass

def check_reward(i,lvl='racetrackcircle.pk'):
    rew = i[4]
    pos_x = i[0].loc[0]
    pos_y = i[0].loc[1]


    if level_load == 'toughertrack.pk':
        # print('test')
        dist_goal1 = ((g1[0] - pos_x) ** 2 + (g1[1] - pos_y) ** 2) ** .5
        dist_goal2 = ((g2[0] - pos_x) ** 2 + (g2[1] - pos_y) ** 2) ** .5
        dist_goal3 = ((g3[0] - pos_x) ** 2 + (g3[1] - pos_y) ** 2) ** .5
        # print(g3)
        dist_goal4 = ((g4[0] - pos_x) ** 2 + (g4[1] - pos_y) ** 2) ** .5
        dist_goal5 = ((g5[0] - pos_x) ** 2 + (g5[1] - pos_y) ** 2) ** .5
        # if rew==0:
        if rew > 1.92:
            rew = 2 +(1000-dist_goal5) / 1000
        elif rew > .92:
            rew = 1 + (1000 - dist_goal4) / 1000  # print(rew)
        else:
            rew = 0+ (1000-dist_goal3)/1000




    elif level_load =='racetrackcircle.pk':
        x=quadrant(i)

        # last_x = last_pos[0]
        # last_y = last_pos[1]

        dist_goal4 = ((quad4to1[0] - pos_x) ** 2 + (quad4to1[1] - pos_y) ** 2) ** .5
        dist_goal3 = ((quad3to4[0]-pos_x)**2+(quad3to4[1]-pos_y)**2)**.5
        dist_goal2 = ((quad2to3[0] - pos_x) ** 2 + (quad2to3[1] - pos_y) ** 2) ** .5
        dist_goal1 = ((quad1to2[0] - pos_x) ** 2 + (quad1to2[1] - pos_y) ** 2) ** .5
        # screen.blit(pygame.font.Font(None, 25).render('x', True, (255, 0, 0)), (500,700))
        if (x == 1) & (rew<=1) :
            rew = 0 + (1000 - dist_goal1) / 1000

        # elif (x==1) & (pos_x > (x_quad-50)):
        #     rew=1
        elif rew>=6:
            rew=6+ (1000-dist_goal1)/1000 + (1000-pos_y)/1000
        elif (x == 1) & (rew>4) :
            rew = 6
        elif (x == 2) :
            # if max([hist[2] for hist in i[6]]) > .1:
            if False:
                rew = 1 + (1000 - dist_goal2) / 1000 +  max([hist[2] for hist in i[6]][0])/2

            else:
                rew = 1 + (1000 - dist_goal2) / 1000
        elif (x == 4) & (rew>1) :
            rew = 4+ (1000-dist_goal4)/1000
        # elif (x == 3) & (pos_x < (x_quad + 50)) & (rew >= 2):
        #     rew = 3
        elif (x == 3) & (rew>1) :
            if max([hist[0] for hist in i[6]]) > .5:
                rew = 2 + (1000 - dist_goal3) / 1000 + .6
            else:
                rew = 2 + (1000 - dist_goal3) / 1000
        elif (rew >= 2):
            # rew = max(rew, 2 + (1000 - pos_x) / 1000)
            # rew = max(rew, 2+ (1000-dist_goal3)/1000)
            rew =  2 + (1000 - dist_goal3) / 1000
        elif (x==2) & (pos_y>(y_quad-50)) & (rew>=1) :
            rew=2

    return rew

def go_right(bot):
    use=None
    if level_load == 'toughertrack.pk':
        if bot[4]<1:
            use = True
        elif bot[4]>1:
            use = False
        elif bot[4]>2:
            use = True
    else:
        use = False

    return use
        
def dist_to_goals(i):

    pos_x = i[0].loc[0]
    pos_y = i[0].loc[1]
    xx= quadrant(i)
    dist_goal4 = ((quad4to1[0] - pos_x) ** 2 + (quad4to1[1] - pos_y) ** 2) ** .5
    dist_goal3 = ((quad3to4[0] - pos_x) ** 2 + (quad3to4[1] - pos_y) ** 2) ** .5
    dist_goal2 = ((quad2to3[0] - pos_x) ** 2 + (quad2to3[1] - pos_y) ** 2) ** .5
    dist_goal1 = ((quad1to2[0] - pos_x) ** 2 + (quad1to2[1] - pos_y) ** 2) ** .5

    if xx==1 :
        dist_goal2 *=0
        dist_goal3  *=0
        dist_goal4 *=0
        dist_goal1 *=1
    elif xx==2:
        dist_goal2 *= 1
        dist_goal3 *= 0
        dist_goal4 *= 0
        dist_goal1 *= 0
    elif xx==3:
        dist_goal2 *= 0
        dist_goal3 *= 1
        dist_goal4 *= 0
        dist_goal1 *= 0
    elif xx==4:
        dist_goal2 *= 0
        dist_goal3 *= 0
        dist_goal4 *= 1
        dist_goal1 *= 0
    return dist_goal1,dist_goal2,dist_goal3,dist_goal4

you_lose=0
pos_data=[]

action_data=[]
array_dist_x=[]
array_dist_y=[]
array_dist_xy=[]
g1 = [150, 300] #goal for level_load = "toughertrack.pk"
g2 = [350, 200] #goal for level_load = "toughertrack.pk"
g3 = [800, 300] #goal for level_load = "toughertrack.pk"
g4 = [600, 800] #goal for level_load = "toughertrack.pk"
g5 = [900, 900] #goal for level_load = "toughertrack.pk"
boxOb = []
bot=[]

botNNet=[]


# yarr gen1 = reset last best & learn, gen 2= no random just learn , gen 3 = all stochastic no learn
gen =2
mut_pct = 0.15
mut_amt1 = 0.45
# sim2_mut_pct = [50,.35] #at x sim switch to 2nd mut pct
# sim3_mut_pct = [100,.55] #at x sim switch to 3nd mut pct

num_bots = 100
diagnostic_move = False
use_cut_logic = False
h_input = 64
sim_count = 500
end_simulation_bot_move = 350

cycle_through_neurons = False
if cycle_through_neurons:
    hidden_layer_sim_use =[[50,2],[100,4],[150,8],[200,16],[250,32],[300,64],[350,128],[400,256],[450,512]]
    # hidden_layer_sim_use =[[50,4],[100,4],[150,4],[200,4],[250,4],[300,4],[350,4],[400,4],[450,4]]
    # hidden_layer_sim_use = [[50, 512], [100, 512], [150, 512], [200, 512], [250, 512], [300, 512], [350, 512], [400, 512], [450, 512]]
    # hidden_layer_sim_use = [[50, 4], [100, 4], [150, 4], [200, 4], [250, 4], [300, 64], [350, 64], [400, 64],
    hidden_layer_sim_use = [[50, 64], [100, 64], [150, 64], [200, 64], [250, 64], [300, 64], [350, 64], [400, 64], [450, 64]]

if num_bots == 1 : run_one_bot = True;
else: run_one_bot= False;

Cross_type = 'one_point' #'one_point / three_point / all_point'
# cross_type_to_use = 'best_worst' # best_best best_worst best_avg best_rand
sim2_cross_type = [66,'three_point'] #at x sim switch to 2nd cross type
sim3_cross_type = [132,'all_point'] #at x sim switch to 3rd cross type
reset_best = False
all_or_nothing_output_neuron_fire = True

x_input_param = numpy.random.random_integers(0,1,size=(14))
asexual_reproduce = True
random_starts_y_axis = False
cut_if_more_bots_than_this = 30
cross_over_test = True
quad1to2 = [500,450]
quad2to3 = [800,600]
quad3to4 = [500,750]
quad4to1 = [200,600]

total_move_probs =[]

start_gene =[]
cut_freq=0
dist_near = (10,100)
run_ai_bool = True
Turn_off_level_create = run_ai_bool 
# level_load = "racetrack.pk"
# level_load = "racetrack2.pk"
# level_load = "racetrack3.pk"
# level_load = "racetrackcircle.pk"
level_load = "toughertrack.pk"

x_quad=500
y_quad=600
sim = 0

start_gene_parents =[]
rew = 0
rew_array=[]
last_pos=[[100,500]]
sav_gen=[]


start_bots = Generate_nextgen()
Recombination(start_bots[0],start_bots[1])

       
x_tmp1 ={1:[True,False,False,False],2:[False,True,False,False],3:[False,False,True,False],4:[False,False,False,True]}

keep_right=False
keep_moving = False
ix=0
iy=0
# ix2=0
cut=0
improvement=[]
improvement_avg=[]
screen = pygame.display.set_mode((1000,1000))
dst=[]
hit=[]
max_x=[]
turn_off_del_logic= [False,False,False,False]
bot_move=1
if run_ai_bool: load_game(), print('loaded');
count_bot = num_bots
reinitialize_simulation = False
end_simulation = False


if not Turn_off_level_create:
    f = open(level_load, 'rb')
    up = pickle.Unpickler(f)
    boxOb = up.load()
    f.close()
while not game_over:
    # rerun simulation
    if reinitialize_simulation & (sim<sim_count):
        if cycle_through_neurons:
            xx1 = [i[0] for i in hidden_layer_sim_use]
            # print(xx1)
            gen = 1
            reset_best= False
            for i in xx1:
                if sim==i:
                    gen=3
                    reset_best = True
                    zztop = xx1.index(i)
                    # print(zztop)
                    h_input = hidden_layer_sim_use[zztop][1]
                    # print(h_input)

        # if sim==sim2_cross_type[0]:
        #     Cross_type = sim2_cross_type[1]
        # elif sim==sim3_cross_type[0]:
        #     Cross_type =sim3_cross_type[1]
        # # elif sim==150
        #     Cross_type = 'all_point'

        sim+=1
        start_gene.clear()
        sav_gen.clear()
        next_bots = Generate_nextgen()
        Recombination(next_bots[0],next_bots[1])
        # print(len(bot))
        #
        # if gen==1 or gen==2:
        #     savename = "genes.pk";f = open(savename, 'rb');up = pickle.Unpickler(f);start_gene = up.load();f.close()
        #     savename = "genes_cross.pk";f = open(savename, 'rb');up = pickle.Unpickler(f);start_gene_parents = up.load();f.close()
        #     sav_gen.append((start_gene[2][1],start_gene[2][2]))
        #     bot.clear()

            # for i in range(num_bots):
            #     # cross_over here too
            #     # Generate_mutants()
            #     # print(start_gene_parents)
            #     # print(start_gene_parents[1],start_gene_parents[0])
            #
            #     Generate_children(start_gene_parents[0],start_gene_parents[1])
# =============================================================================
#                 yy= Nnet1().mutate(sav_gen[0][0],sav_gen[0][1],mut_pct,mut_pct)
#                 bot.append([Player([100,500]),Nnet1(),yy,you_lose,rew,last_pos])
# =============================================================================
#         elif gen==3:
#             for i in range(num_bots):
#                 last_pos.clear()
#                 bot.append([Player([100,500]),Nnet1(),Nnet1().forward(x_input_param),you_lose,rew,last_pos])
        ix=0
        iy=0
        # ix2=0
        cut=0
        turn_off_del_logic= [False,False,False,False]
        bot_move=1
        # count_bot = num_bots
        reinitialize_simulation = False
        # end restart simulation
        rew_array.clear()
        
    
    # when to end simulation?
    if (bot_move >end_simulation_bot_move) or (len(bot)==0):
        # print('end now')
        end_simulation = True
    
    ix+=1
    # ix2+=1
    iy+=1
    # print(iy)
    

    speed=1
            
        
    
    if (iy>speed) & run_ai_bool :
        iy=0
        if ix==100000: ix=0;
        
        count_bot = len(bot)
        cut+=cut_freq
        bot_move+=1
        for i in reversed(bot):
            # i[4] = check_reward(i[4],i[0].loc[0],i[0].loc[1],last_pos[0],last_pos[1])
            # print('checking')
            # print(i[4])
            i[4] = check_reward(i)
            # print('checking1')

# =============================================================================
#             if ((ix%100)==0):
#                 print('checking')
#                 i[4] = check_reward(i[4],i[0].loc[0],i[0].loc[1])
# =============================================================================

            # line_sight = line_of_sight(i)
            line_sight = collission_box_and_line_sight(i)[:]
             # = temp_coll_line[:]
            coll_ = line_sight[4]

            temp_coll_diag = diag_line(i)[:]
            quad_use = x_tmp1[quadrant(i)]
            dist_near = dist_to_goals(i)
            last_pos = i[5]

            if len(last_pos)<1: last_pos.append([100,500]);
                # momentum =[0,0]
            # print(last_pos,last_pos[-1], i[0].loc[0])
            momentum = [i[0].loc[0]-last_pos[-1][0],i[0].loc[1]-last_pos[-1][1]]
            # print(type(i[0].loc[0]\
            #            ,last_pos[0]))
            x_val = [ \
                 i[0].loc[0], i[0].loc[1] \
                 ,line_sight[0], line_sight[1], line_sight[2], line_sight[3]  \
                  ,line_sight[5], line_sight[6], line_sight[7], line_sight[8] \
                  # ,quad_use[0] * 1000, quad_use[1] * 1000, quad_use[2] * 1000, quad_use[3] * 1000 \
                 # ,dist_near[0] * 10, dist_near[1] * 10, dist_near[2] * 10 \
                , momentum[0], momentum[1] \
                ,i[4] \
                ,go_right(i)
                ]

            # x_val = [\
            #     i[0].loc[0],i[0].loc[1],\
            #          line_sight[0],line_sight[1],line_sight[2],line_sight[3] \
            #         ,line_sight[5], line_sight[6], line_sight[7], line_sight[8] \
            #         ,quad_use[0]*1000,quad_use[1]*1000,quad_use[2]*1000,quad_use[3]*1000\
            #         ,dist_near[0]*10,dist_near[1]*10,dist_near[2]*10\
            #         ,momentum[0],momentum[1]\
            #         ]

            i[5].append([i[0].loc[0], i[0].loc[1]])
            # print(last_pos)
            run_ai2(x_val,i);
            if run_one_bot:
                clock.tick(3)
                print('right,left,down,up,quad1,quad2,quad3,quad4')
                print(x_val)
                # print(run_ai2)


            # print(last_pos,last_pos[0])
# =============================================================================
#             if collission_box(i) or (i[0].loc[0]< cut):
#                 bot.remove(i)       
# =============================================================================
            # print(ix)
            if coll_  :
                # i[3]+=-100
                max_x.append(i[0].loc[0])
                # logic added to disuade dying
                tmp_x=i
                # tmp_x[4]+=-0.5
                tmp_x[4] += -0.01
                rew_array.append(tmp_x)

                bot.remove(i)
# =============================================================================
#         remove_1 = [x for x in bot if collission_box_and_line_sight(x)[4]]
#         if len(remove_1) > 0 : bot.remove(remove_1)   ;
# =============================================================================

    # delete bot if they go down into quad 4 right away
    if False:
        for ixy in reversed(bot):
            # print(ixy[4])
            # print(ixy[4])
            # print('yarr:',len(bot))
            if (quadrant(ixy)==4) &(ixy[4]<=3):
                rew_array.append(ixy)
                bot.remove(ixy)

    if (len(bot)>cut_if_more_bots_than_this) & (use_cut_logic):
        if (bot_move>8) & (not turn_off_del_logic[0]):
            print('cut 1')
            for ixy in reversed(bot):
                if (ixy[0].loc[1]>500) or (ixy[0].loc[0]<100):
                    rew_array.append(ixy)
                    bot.remove(ixy)
            turn_off_del_logic[0] = True
            
        elif (bot_move>15) & (not turn_off_del_logic[3]): 
            print('cut 2')
            for ixy in reversed(bot):
                if (ixy[0].loc[1]>500) or (ixy[0].loc[0]<100):
                    rew_array.append(ixy)
                    bot.remove(ixy)
            turn_off_del_logic[3] = True
            
        elif (bot_move>60) & (not turn_off_del_logic[1]): 
            print(bot_move,'teminating if not reached reward 1/gate1')
            for ixy in reversed(bot):
                if ixy[4]<1:
                    rew_array.append(ixy)
                    bot.remove(ixy)
            turn_off_del_logic[1] = True
        
        elif (bot_move>180) & (not turn_off_del_logic[2]): 
            # print('test')
            print(bot_move,'teminating if not reached reward 3/gate3')
            for ixy in reversed(bot):
                if ixy[4]<3:
                    rew_array.append(ixy)
                    bot.remove(ixy)
            turn_off_del_logic[2] = True

    if (not Turn_off_level_create ):
        if (collission_box_and_line_sight(bot[0])[4]):
            print('gameover')
    for event in pygame.event.get():
        # print(pygame.mouse.get_pos())
        if event.type == pygame.QUIT:
            game_over=True
        if not Turn_off_level_create:
            if (event.type == pygame.KEYDOWN) :
                for i in reversed(bot):
                    # i[4] = check_reward(i)
                    # =============================================================================
                    #             if ((ix%100)==0):
                    #                 print('checking')
                    #                 i[4] = check_reward(i[4],i[0].loc[0],i[0].loc[1])
                    # =============================================================================

                    # line_sight = line_of_sight(i)
                    # line_sight = collission_box_and_line_sight(i)[:]
                    # # = temp_coll_line[:]
                    # coll_ = line_sight[4]
                    #
                    # temp_coll_diag = diag_line(i)[:]
                    # quad_use = x_tmp1[quadrant(i)]
                    i[4] = check_reward(i)
                    # print(i[4])
                    # print('checking1')

                    # =============================================================================
                    #             if ((ix%100)==0):
                    #                 print('checking')
                    #                 i[4] = check_reward(i[4],i[0].loc[0],i[0].loc[1])
                    # =============================================================================

                    # line_sight = line_of_sight(i)
                    line_sight = collission_box_and_line_sight(i)[:]
                    # = temp_coll_line[:]
                    coll_ = line_sight[4]

                    temp_coll_diag = diag_line(i)[:]
                    quad_use = x_tmp1[quadrant(i)]
                    dist_near = dist_to_goals(i)
                    last_pos = i[5]

                    if len(last_pos) < 1: last_pos.append([100, 500]);
                    # momentum =[0,0]
                    # print(last_pos,last_pos[-1], i[0].loc[0])
                    momentum = [i[0].loc[0] - last_pos[-1][0], i[0].loc[1] - last_pos[-1][1]]

                    x_val =  [\
                  i[0].loc[0], i[0].loc[1] \
                 ,line_sight[0], line_sight[1], line_sight[2], line_sight[3]  \
                  ,line_sight[5], line_sight[6], line_sight[7], line_sight[8] \
                   ,quad_use[0] * 1000, quad_use[1] * 1000, quad_use[2] * 1000, quad_use[3] * 1000 \
                  ,dist_near[0] * 10, dist_near[1] * 10, dist_near[2] * 10 \
                 , momentum[0], momentum[1] \
                        ]
                    xxx= i[1].lockin(x_val, i[2][1], i[2][2])
                    xxx= numpy.array(xxx).round(2).reshape(1,4)
                    xxy = xxx.argmax() + 1
                    print('\nleft/up/down/right\n',xxx,xxy)
                    # for i in xxx:
                    #     print('{} '.format(i))
                    # x_val = [ \
                    # # i[0].loc[0],i[0].loc[1],\
                    # line_sight[0], line_sight[1], line_sight[2], line_sight[3] \
                    # , line_sight[5], line_sight[6], line_sight[7], line_sight[8] \
                    # # ,quadrant(i)*100\
                    # , quad_use[0] * 1000, quad_use[1] * 1000, quad_use[2] * 1000, quad_use[3] * 1000 ]
                    #     # x_val = [pl.loc[:][0],pl.loc[:][1],you_lose]

                # keep_going_key= True
                # print(clf.predict([pl.loc][:])[0])
                print(x_val)
                # print('reward: {:.2f}'.format(i[4]))
                if coll_:
                    print('gameover')
                if (event.key == pygame.K_RIGHT) :
                    # track_data(x_val)
                    # action_data.append((False,False,False,True))
                    # pl.loc[0]+=10
                    # print(bot[0][0])
                    bot[0][0].loc[0] += 10
                elif event.key == pygame.K_UP:
                    # pl.loc[1]+=-10
                    # track_data(x_val)
                    # action_data.append((False,True,False,False))
                    # print(pl.loc)
                    # print((False,True,False,False))
                    bot[0][0].loc[1] += -10
                elif event.key == pygame.K_DOWN:
                    # pl.loc[1]+=+10
                    # track_data(x_val)
                    # action_data.append((False,False,True,False))
                    bot[0][0].loc[1] += +10
                elif event.key == pygame.K_LEFT:
                    # pl.loc[0]+=-10
                    # track_data(x_val)
                    # action_data.append((True,False,False,False))
                    bot[0][0].loc[0] += -10
                elif  event.key == pygame.K_INSERT:
                    # savename = "racetrackcircle.pk"
                    savename = "toughertrack.pk"
                    f = open(savename, 'wb')
                    p = pickle.Pickler(f)
                    p.dump(boxOb)
                    f.close()
                elif event.key == pygame.K_BACKQUOTE:
                    # level_load = "racetrack.pk"
                    f = open(level_load, 'rb')
                    up = pickle.Unpickler(f)
                    boxOb = up.load()
                    f.close()
                elif event.key == pygame.K_PAGEDOWN:
                    f = open('datagame.pk', 'wb'); p = pickle.Pickler(f);p.dump([pos_data,action_data]);f.close()
                elif event.key == pygame.K_KP_ENTER:
                    savename = "genes.pk"
                    f = open(savename, 'wb')
                    p = pickle.Pickler(f)
                    p.dump(bot)
                    f.close()
        # if collission_box():
        #     you_lose=True
        if (event.type == pygame.MOUSEBUTTONDOWN) & (pygame.mouse.get_pressed() == (0,1,0)):    
              mouse = pygame.mouse.get_pos() 
              # print('try to save this one')
              for boxy in bot:
                  if (boxy[0].loc[0] < pygame.mouse.get_pos()[0] < boxy[0].loc[0]+50 ) \
                      & (boxy[0].loc[1] < pygame.mouse.get_pos()[1] < boxy[0].loc[1]+50):
                    print('save this one')
                    start_gene = boxy
                    savename = "genes.pk"
                    f = open(savename, 'wb')
                    p = pickle.Pickler(f)
                    # p.dump()
                    p.dump(start_gene)
                    f.close()
        # if (event.type == pygame.MOUSEBUTTONDOWN) & (pygame.mouse.get_pressed() == (1,0,0)) & Turn_off_level_create:    
        if (pygame.mouse.get_pressed() == (1,0,0)) & Turn_off_level_create:    
              mouse = pygame.mouse.get_pos() 
              # print('try to save this one')
              for boxy in bot:
                  if (boxy[0].loc[0] < pygame.mouse.get_pos()[0] < boxy[0].loc[0]+50 ) \
                      & (boxy[0].loc[1] < pygame.mouse.get_pos()[1] < boxy[0].loc[1]+50):
                          bot.remove(boxy)
                    
        if not Turn_off_level_create:
            if (event.type == pygame.MOUSEBUTTONDOWN) & (pygame.mouse.get_pressed() == (0,0,1)):
                mouse = pygame.mouse.get_pos() 
                boxOb.append(BoxPipeData(list(mouse[:]),[0,0],[],10,'name',(150,150,150),50))
            
         
            if keep_moving:
                move_box(box_move)
                if (pygame.mouse.get_pressed() == (0,0,0)):
                    keep_moving = False        
            
    #         Left Click to move boxes
            for boxy in boxOb:
                if (pygame.mouse.get_pressed() == (1,0,0)) & (boxy.loc[0] < pygame.mouse.get_pos()[0] < boxy.loc[0]+size[0] ) \
                    & (boxy.loc[1] < pygame.mouse.get_pos()[1] < boxy.loc[1]+size[1] ):
                        keep_moving = True
                        box_move=boxy
                        # boxy.color
                        to_write=boxy
            # delete boxes
            for boxy in boxOb:
                if (pygame.mouse.get_pressed() == (1,1,0)) & (boxy.loc[0] < pygame.mouse.get_pos()[0] < boxy.loc[0]+size[0] ) \
                    & (boxy.loc[1] < pygame.mouse.get_pos()[1] < boxy.loc[1]+size[1] ):
                        boxOb.remove(boxy)

  
    screen.fill((0,0,0))

    draw_boxes()
    draw_players()
    # txt = 'mut:' + str(round(mut_pct,2)) +
    to_disp = 'max rew:'+str(round(max([x[4] for x in bot]+[0]),3))
    to_disp2 =  'bot move:' + str(bot_move)
    to_disp3 = 'bots alive:' + str(len(bot))
    screen.blit(pygame.font.Font(None, 25).render(to_disp, True, (255, 255, 255)), [0,0])
    screen.blit(pygame.font.Font(None, 25).render(to_disp2, True, (255, 255, 255)), [0, 15])
    screen.blit(pygame.font.Font(None, 25).render(to_disp3, True, (255, 255, 255)), [0, 30])
    if level_load == "racetrackcircle.pk":
        pygame.draw.line(screen,(255,255,255),[x_quad,0],[x_quad,1000],1)
        pygame.draw.line(screen,(255,255,255),[0,y_quad],[1000,y_quad],1)
        screen.blit(pygame.font.Font(None, 25).render('x', True, (255, 0, 0)), quad1to2)
        screen.blit(pygame.font.Font(None, 25).render('x', True, (255, 0, 0)), quad2to3)
        screen.blit(pygame.font.Font(None, 25).render('x', True, (255, 0, 0)), quad3to4)
        screen.blit(pygame.font.Font(None, 25).render('x', True, (255, 0, 0)), quad4to1)
    if level_load == "toughertrack.pk":
        # pygame.draw.line(screen, (255, 255, 255), [x_quad, 0], [x_quad, 1000], 1)
        # pygame.draw.line(screen, (255, 255, 255), [0, y_quad], [1000, y_quad], 1)
        screen.blit(pygame.font.Font(None, 25).render('x', True, (255, 0, 0)), g1)
        screen.blit(pygame.font.Font(None, 25).render('x', True, (255, 0, 0)), g2)
        screen.blit(pygame.font.Font(None, 25).render('x', True, (255, 0, 0)), g3)
        screen.blit(pygame.font.Font(None, 25).render('x', True, (255, 0, 0)), g4)
        screen.blit(pygame.font.Font(None, 25).render('x', True, (255, 0, 0)), g5)
    pygame.display.update()

    if end_simulation:

        print('total probs moving: left,up,down,right')
        print(len(total_move_probs))
        total_move_probs = numpy.reshape(total_move_probs,(len(total_move_probs),4))
        print(total_move_probs)
        total_move_probs =[]

        live_bot_rewards=[i for i in bot]
        rew_array.extend(live_bot_rewards)
        xij = [i[4] for i in rew_array]
        max_rew = max(xij)
        sort_list_best = sorted(rew_array, key=lambda x: x[4], reverse=True)
        prnt_fitness = [round(i[4],2) for i in sort_list_best]
        print(prnt_fitness)
        # print('{:.2f}'.format(*prnt_fitness))
        print('max_reward:{}'.format(round(max_rew),2))
        print('Avg fitness of pop: {:.3f}'.format(sum(xij)/len(xij)))
        
        savename = "genes.pk";f = open(savename, 'rb');up = pickle.Unpickler(f);
        load_last_best = up.load();f.close();
        savename = "genes_cross.pk";f = open(savename, 'rb');up = pickle.Unpickler(f);
        load_last_best2 = up.load();f.close();
        # load_last_best = 
        print('last:{} newbest:{}'.format(round(load_last_best[4],2),round(sort_list_best[0][4],2)))
        # print(sort_list_best[0][0].loc[0],sort_list_best[0][0].loc[1])

        zz=sort_list_best[0][6]
        zz=numpy.reshape(zz,(len(zz),4))
        zz1=max([i[0] for i in zz])
        zz2 = max([i[1] for i in zz])
        zz3 = max([i[2] for i in zz])
        zz4 = max([i[3] for i in zz])
        # print(zz)
        print('\nleft,up,down,right\n',zz1,zz2,zz3,zz4)

        if ((gen!=2) & (sim==0)) or (reset_best):
            print('reset')
            load_last_best[4] = -0.6
            load_last_best2[0][4] = -0.6
            
        # if cross_over_test & (load_last_best2[0][4]<sort_list_best[0][4]):
        #     print('in')
        #     save_best_genes_parents(sort_list_best,num=2)
        if False:
            # print('in')
            # save_best_genes_parents(sort_list_best)
            print('new genes even if they are worse')
            save_all_parents_curr_gen(sort_list_best)
        if False:
            print('one off save the best_so_far.pk')
            save_all_parents_curr_gen(sort_list_best,savenamestr='best_so_far.pk')
        if(load_last_best[4] < sort_list_best[0][4]):
            print('new genes')
            save_best_genes_mutants(sort_list_best[0])
            save_all_parents_curr_gen(sort_list_best)
        else:
            pass
        reinitialize_simulation = True
        end_simulation= False
        bot.clear()
        improvement.append(round(max_rew,2))
        improvement_avg.append(round(sum(xij)/len(xij), 2))
        print('sim {} of {}\n'.format(sim, sim_count))
        # print('\n#inputs:{} #botmove:{} #hiddenneurons:{} num_bots:{} mutation{} crosstype:{}'.format(len(x_input_param),bot_move,Nnet1.h_output,num_bots,mut_pct,Cross_type))
        print(
            '\n#inputs:{} #botmove:{} #hiddenneurons:{} num_bots:{} mut_chance{} mut_amount {} crosstype:{}'.format(len(x_input_param),
                                                                                                    bot_move,
                                                                                                    h_input,
                                                                                                    num_bots, mut_pct,mut_amt1,
                                                                                                    Cross_type))
        print('\nmax over time:{} \navg pop max{}'.format(improvement,improvement_avg))
        print('=============================================================================\
              =============================================================================')
        if sim==sim_count:
            game_over=True; 
            pygame.quit();

pygame.quit()