import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import KFold, cross_val_score, cross_validate, train_test_split, StratifiedKFold, cross_val_predict 
from sklearn.metrics import mean_absolute_error, mean_squared_error


 # ============================== 1_Basic_Exploration ============================== #
    
    
def histogram(df=None):
    fig = make_subplots(rows=4, cols=2, specs=[[{"colspan": 1}, {"colspan": 1}],
                                              [{"colspan": 1}, {"colspan": 1}],
                                              [{"colspan": 1}, {"colspan": 1}],
                                              [{"colspan": 1}, {"colspan": 1}]],
                        subplot_titles=('cylinders', 'displacement', 'horsepower', 'weight', 'acceleration',
                                       'model year', 'origin', 'mpg'))

    fig.add_trace(go.Histogram(x=df["cylinders"], name="cylinders", nbinsx=35, showlegend=True, legendgroup='1'), 
                  row=1, col=1)
    fig.add_trace(go.Histogram(x=df["displacement"], name="displacement", nbinsx=35, showlegend=True, legendgroup='1'),
                  row=1, col=2)
    fig.add_trace(go.Histogram(x=df["horsepower"], name="horsepower", nbinsx=35, showlegend=True,
                  legendgroup='1'), row=2, col=1)
    fig.add_trace(go.Histogram(x=df["weight"], name="weight", nbinsx=35, showlegend=True, legendgroup='1'),
                  row=2, col=2)
    
    fig.add_trace(go.Histogram(x=df["acceleration"], name="acceleration", nbinsx=35, showlegend=True,
                   legendgroup='1'), row=3, col=1)
    fig.add_trace(go.Histogram(x=df["model year"], name="model year", nbinsx=35, showlegend=True,
                   legendgroup='1'), row=3, col=2)
    fig.add_trace(go.Histogram(x=df["origin"], name="origin", nbinsx=35, showlegend=True,
                   legendgroup='1'), row=4, col=1)
    fig.add_trace(go.Histogram(x=df["mpg"], name="mpg", nbinsx=35, showlegend=True,
                   legendgroup='1'), row=4, col=2)
        
        
    fig.update_layout(plot_bgcolor='#F8F8F6',
                  height=950, width=950, bargap=0.2,
                  title_text='Histogram of Numerical Features', title_font_size=20, title_font_family='Arial Black',                      
                  title_x=0, title_y=0.98,
                  margin=dict(l=0, r=20, t=130, b=80))

    fig.update_annotations(yshift=20)
    #fig.update_traces(insidetextfont_size=10, selector=dict(type='pie'))
    fig.update_yaxes(title='Count')  
    fig.show()

def boxplot(df=None):
    fig = make_subplots(rows=4, cols=2) # Creating a matrix of empty subplots. 

    # Adding each plot into our matrix. 
    # Column 1. 
    fig.append_trace(go.Box(x=df["cylinders"], name="cylinders"), row=1, col=1)
    fig.append_trace(go.Box(x=df["displacement"],name="displacement"), row=2, col=1)
    fig.append_trace(go.Box(x=df["horsepower"],name="horsepower"), row=3, col=1)
    fig.append_trace(go.Box(x=df["weight"], name="weight"), row=4, col=1)

    # Column 2.
    fig.append_trace(go.Box(x=df["acceleration"], name="acceleration"), row=1, col=2)
    fig.append_trace(go.Box(x=df["model year"],name="model year"), row=2, col=2)
    fig.append_trace(go.Box(x=df["origin"],name="origin"), row=3, col=2)
    fig.append_trace(go.Box(x=df["mpg"], name="mpg"), row=4, col=2)

    # Editing the layout and showing the figure. 
    fig.update_layout(plot_bgcolor='#F8F8F6', title_text="Boxplot of Numerical Features", title_font_size=20, 
                title_font_family='Arial Black', title_x=0, title_y=0.98, height = 650, width = 950)
    fig.show()
    
    
 # ============================== 2_Base_Models ============================== #


def kfold(scores=None):
    fig = go.Figure() # Creating blank figure. 

    # Adding indivdual boxplots to figure. 
    fig.add_trace(go.Box(y=list(scores[1]), name="EN"))
    fig.add_trace(go.Box(y=list(scores[2]), name="CART"))
    fig.add_trace(go.Box(y=list(scores[3]), name="SVR"))
    fig.add_trace(go.Box(y=list(scores[4]), name="KNN"))
    fig.add_trace(go.Box(y=list(scores[5]), name="GBR"))
    fig.add_trace(go.Box(y=list(scores[6]), name="RFR"))
    fig.add_trace(go.Box(y=list(scores[7]), name="XGB"))

    # Configuring layout. 
    fig.update_layout(plot_bgcolor='#F8F8F6',height=450, width=950, title_text="Performance Over 5 Folds",
                      title_font_size=20, title_font_family='Arial Black')
    fig.show() # Showing figure. 

    
  # ============================== 3_Feature_Engineering ============================== #


def plot_rfe(xaxis=None, yGBR=None, yRFR=None, yXGB=None):
    # Creating figure. 
    fig = go.Figure()
    # Adding subplots. 
    fig.add_trace(go.Scatter(x=xaxis, y=yGBR,name='Gradient Boosting Regressor')) 
    fig.add_trace(go.Scatter(x=xaxis, y=yRFR,name='Random Forest Regressor'))
    fig.add_trace(go.Scatter(x=xaxis, y=yXGB,name='XGB Regressor'))
    # Defining labels.  
    fig.update_layout(title='Recursive Feature Elimination for Selected Models',
                       xaxis_title='Total Features Selected',
                       yaxis_title='Mean Squared Error', width=980, height=450, plot_bgcolor='#F8F8F6', title_font_size=20, 
                       title_font_family='Arial Black')
    fig.show() # Display figure. 

    
  # ============================== 4_Optimization ============================== #

def n_components_plot(xaxis=None, yGBR=None, yRFR=None, yXGB=None):
    # Setting up our figure canvas. 
    fig = make_subplots(rows=3, cols=1)

    fig.add_trace(go.Scatter(x=xaxis, y=yGBR, name="Gradient Boosting Regressor"), row=1, col=1)
    fig.add_trace(go.Scatter(x=xaxis, y=yRFR, name="Random Forest Regressor"), row=2, col=1) 
    fig.add_trace(go.Scatter(x=xaxis, y=yXGB, name="XGB Regressor"), row=3, col=1)

    # Axes labels. 
    fig['layout']['xaxis3']['title']='n_components'
    fig['layout']['yaxis']['title']='MSE'

    fig.update_layout(title_text="Finding Optimal N Components", height=550, width=800, plot_bgcolor='#F8F8F6', 
                      title_font_size=20, title_font_family='Arial Black') # Configure dimensions.
    fig.show() # Display plots. 
        
def grid_search_plot(instance, model_string):
    MSE = abs(instance.grid_results.cv_results_['mean_test_score'])
    iterations2 = list()
    [iterations2.append(i) for i in range(1,len(MSE)+1)];

    # Plot early stopping results. 
    fig = px.line(x=iterations2, y=MSE)
    # Best loss score. 
    fig.add_vline(x=list(MSE).index(min(MSE))+1, line_width=2, line_dash="dash", line_color="black")  
    fig.update_layout(title='Hypertuning the {}'.format(model_string),
                           xaxis_title='No. of Iterations',
                           yaxis_title='MSE', height=400, width=800, plot_bgcolor='#F8F8F6', title_font_size=20, 
                       title_font_family='Arial Black')
    
    fig.show()

    
 # ============================== 5_Regularization ============================== #        
    
    
def LogLoss_Curve(x_axis, y_axis_train, y_axis_test):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(x_axis), y=y_axis_train,
                        name='Train'))
    fig.add_trace(go.Scatter(x=list(x_axis), y=y_axis_test,
                        name='Test'))

    fig.update_layout(title='XGB Regressor Loss Curve',
                       xaxis_title='Epochs',
                       yaxis_title='MAE',
                       height=400, width=800, plot_bgcolor='#F8F8F6', title_font_size=20, 
                       title_font_family='Arial Black')
    fig.show() 


 # ============================== 6_Final_Pipeline ============================== #
    

def MSE_plot(MSE_scores_GBR, MSE_scores_RFR, MSE_scores_XGB, score_string):
    fig = make_subplots(rows=3, cols=1)

    fig.append_trace(go.Scatter(
        x=list(range(1,31)),
        y=MSE_scores_GBR,
        name="Gradient Boosting Regressor"
    ), row=1, col=1)

    fig.append_trace(go.Scatter(
        x=list(range(1,31)),
        y=MSE_scores_RFR,
        name="Random Forest Regressor"
    ), row=2, col=1)

    fig.append_trace(go.Scatter(
        x=list(range(1,31)),
        y=MSE_scores_XGB,
        name="XGB Regressor",
    ), row=3, col=1)


    fig.update_layout(title_text="Model Performance over 30 Iterations", height=550, width=900, plot_bgcolor='#F8F8F6', title_font_size=20, 
                       title_font_family='Arial Black')

    fig['layout']['yaxis']['title']=score_string
    fig['layout']['xaxis3']['title']='Iterations'

    fig.show()
    
    
    
    
    
    
    
    
    