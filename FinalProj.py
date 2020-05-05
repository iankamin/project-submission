from   tkinter import *
import numpy as np
import pandas as pd
#from pandasp.pandas import Table
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import normalize
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)




from scipy import signal

def CSV_to_PLT():
    global data,columns,df,clasY,testY,dfs
    df = pd.read_csv('test.csv')
    columns=df.columns
    
    time = df['Time'].to_numpy().reshape(-1,1)
    clas = df['Class'].to_numpy().reshape(-1,1)

    df1 = df.drop(['Time'], axis=1)
    df1 = df1.drop(['Class'], axis=1)
    #df1=df1.rolling(25).mean()

    data = df1.to_numpy()
    print(data.shape)
    testY=data
    #testY= tester(data)
#    ss = StandardScaler()
    print(data.shape)

#    testY = ss.fit_transform(data)
    clasY = clas
#    data = ss.fit_transform(data)

    #columns = np.array(df.columns)
    #print(columns)

    #data = removeOutliers(data)
    print(data.shape)
    print(time.shape)


    print(data.shape)

    mms = MinMaxScaler()
    dataScaled = mms.fit_transform(data)
    
    data = np.append(time,data, axis=1)

    print(clas.shape)
    #data = np.append(data,clas, axis=1)
    columns[0:15]
    columns=np.append(columns[0:65],['Class'])

    df = pd.DataFrame(data=data, columns=columns)

    dataScaled = np.append(time,dataScaled, axis=1)
    dfs = pd.DataFrame(data=dataScaled, columns=columns)
    

    print(data)

def createGraph(pos):
    global data,columns,dfs
    pos = int(pos)
    fig = plt.figure(figsize=(5,5), dpi=100)
    ax = fig.add_subplot(111, projection="My_Axes")
    print(pos)

    #ax.margins(x=-.4)
    plt.ylim(0,1)
    plt.xlim(0,w)
    print(dfs.columns[0])

    for c,col in enumerate(dfs.columns[1:]):
        print(col)
        dfs.plot(kind='line',x=dfs.columns[0],y=col, ax=ax)

    #fig.show()
    fig.tight_layout()
    #fig.canvas.draw_idle()
    ax.legend(loc='upper left')

    return fig,ax

def tester(data):
    ica = FastICA(n_components=14)
    data = ica.fit_transform(data)
    return data

#def animate

class My_Axes(matplotlib.axes.Axes):
    name = "My_Axes"
    def drag_pan(self, button, key, x, y):
        matplotlib.axes.Axes.drag_pan(self, button, 'x', x, y) # pretend key=='x
        


def onclick(event):
    global line, topleft,ix,iy,curVal
    if pause:
        ix, iy = event.xdata, event.ydata
        line.figure.canvas.draw()
    
line=None
pause=True
w=100
ix=45
iy=0
interval=100
dim=[0,w]
def animate(i):
    global line,ix, curval, ax, fig,Csize
    if not pause:
        ix+=interval
        if (ix)>Csize:
            ix=0
        line.set_xdata(ix)
        curVal.set(currentValues(int(ix)))
        dim[0]=ix-(w/2)
        dim[1]=ix+(w/2)
        plt.xlim(dim[0],dim[1])
        ax.figure.canvas.draw()
    else:
        line.set_xdata(ix)
        curVal.set(currentValues(int(ix)))
        plt.xlim(dim[0],dim[1])
        ax.figure.canvas.draw()
    return line,


def currentValues(time):
    global df,predictions,clasY
    result=str(df.iloc[time][:]) 
    result=nth_repl_all(result,'\n',"]ttttt[",8)
    result=result.replace('\n',"            ")
    result=result.replace(']ttttt[','\n')
    result+="\npredict   "+str(predictions[time])+"\n actual:"+str(clasY.ravel()[time])


    predictions
    testY
    return result
        
def togglePause():
    global pause
    pause = not pause

def nth_repl_all(s, sub, repl, nth):
    find = s.find(sub)
    # loop util we find no match
    i = 1
    while find != -1:
        # if i  is equal to nth we found nth matches so replace
        if i == nth:
            s = s[:find]+repl+s[find + len(sub):]
            i = 0
        # find + len(sub) + 1 means we start after the last match
        find = s.find(sub, find + len(sub) + 1)
        i += 1
    return s





from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors

def training():
    df = pd.read_csv('train.csv')
    columns=df.columns
    time = df['Time'].to_numpy().reshape(-1,1)
    clas = df['Class'].to_numpy().reshape(-1,1)

    df1 = df.drop(['Time'], axis=1)
    df1 = df1.drop(['Class'], axis=1)
    #df1=df1.rolling(25).mean()

    train = df1.to_numpy()
    #train = tester(train)
    #ss = StandardScaler()
    #train = ss.fit_transform(train)


    labels = ['Move Both Fists','Imagine Moving Both Fists','Imagine Moving Left Fist','Imagine Moving Right Fist','Left Fist','Rest Eyes Closed','Rest Eyes Open','Right Fist']
    Ydict  = {'Both_Fists': 0,
             'IMGINE_Both_Fists': 1,
             'IMGINE_Left_Fist': 2,
             'IMGINE_Right_Fist': 3,
             'Left_Fist': 4,
             'Rest_Eyes_Closed': 5,
             'Rest_Eyes_Open': 6,
             'Right_Fist': 7}

    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(train,clas.ravel())

    predicted_value=neigh.predict(testY)
    
    print(predicted_value)
    y = [0,1,2,3,4,5,6,7]
    return predicted_value


    
    # colors = ['#FF3333','#FF9933','#FFFF33','#80FF00','#00FFFF','#0080FF','#0000FF','#7F00FF']
    # colors2 = ['#FFCCCC', '#FFE5CC', '#FFFFCC', '#E5FFCC', '#CCFFFF', '#CCE5FF', '#CCCCFF','#FFCCFF']

    
    
    # fig, ax = plt.subplots()
    # plt.rcParams['axes.facecolor'] = 'white'
    

    # for i in range(samp.shape[0]):
    #     for j in range(neighbors):
    #         xx = np.array([neighborplot[i*neighbors+j,0],samp[i,0]]) #neighborplot[i*neighbors+j,0]
    #         yy = np.array([neighborplot[i*neighbors+j,1],samp[i,1]]) #[neighborplot[i*neighbors+j,1]
    #         ax.plot(xx,yy, 'r--',color = 'gray')
    # for i in range(8):
    #     ax.scatter(neighborplot[neighborplot[:,-1] == i,0], neighborplot[neighborplot[:,-1] == i,1], label = labels[i], marker='o', s= 100, color=colors[i], edgecolor = 'gray')
    # for i in range(samp.shape[0]):
    #     ax.scatter(samp[i,0], samp[i,1], marker='*', s= 750, color='black') 
    
    # ax.legend()
    # plt.show()





pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)







































matplotlib.projections.register_projection(My_Axes)
data=None
columns=None

screen =Tk()
screen.state('zoomed')
root=screen
screen.title("Project")

CSV_to_PLT()
fig, ax = createGraph(0)
line = ax.axvline(x=10, color = 'r')
Csize=data.shape[0]

predictions=training()

#figure = createGraph(barpos)
#fig.show()


left = Frame(screen, borderwidth=2, relief="solid")
topleft = Frame(left, borderwidth=2, relief="solid")

curVal=StringVar()
dataValues = Label(topleft,textvariable=curVal)
dataValues.pack()
curVal.set(currentValues(0))
#pt = (topleft, dataframe=currentValues(0))

button = Button(topleft, 
                   text="PLAY\nPause", 
                   fg="red",
                   command=togglePause)
button.pack(side="right")

left.pack(side="left", expand=True, fill="both")
topleft.pack(expand=True, fill="both", padx=5, pady=5)

canvas = FigureCanvasTkAgg(fig,master=left)
canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
cid = canvas.mpl_connect('button_release_event', onclick)
canvas.draw()
toolbar = NavigationToolbar2Tk(canvas, left)
toolbar.update()
#canvas.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=1)

import matplotlib.animation as animation

ani = animation.FuncAnimation(fig, animate, interval=interval, blit=True, save_count=50)


screen.mainloop()

while 1:
    task.update()


