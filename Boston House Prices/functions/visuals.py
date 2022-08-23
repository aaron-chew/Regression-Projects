import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import KFold, cross_val_score, cross_validate, train_test_split, StratifiedKFold, cross_val_predict 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error


 # ============================== 1_Basic_Exploration ============================== #
    
    
def histogram(df=None):
    fig = make_subplots(rows=4, cols=2, specs=[[{"colspan": 1}, {"colspan": 1}],
                                              [{"colspan": 1}, {"colspan": 1}],
                                              [{"colspan": 1}, {"colspan": 1}],
                                              [{"colspan": 1}, {"colspan": 1}]],
                        subplot_titles=('CRIM', 'ZN', 'LSTAT', 'INDUS', 'CHAS',
                                       'NOX', 'RM', 'MEDV'))

    fig.add_trace(go.Histogram(x=df["CRIM"], name="CRIM", nbinsx=35, showlegend=True, legendgroup='1'), 
                  row=1, col=1)
    fig.add_trace(go.Histogram(x=df["ZN"], name="ZN", nbinsx=35, showlegend=True, legendgroup='1'),
                  row=1, col=2)
    fig.add_trace(go.Histogram(x=df["LSTAT"], name="LSTAT", nbinsx=35, showlegend=True,
                  legendgroup='1'), row=2, col=1)
    fig.add_trace(go.Histogram(x=df["INDUS"], name="INDUS", nbinsx=35, showlegend=True, legendgroup='1'),
                  row=2, col=2)
    
    fig.add_trace(go.Histogram(x=df["CHAS"], name="CHAS", nbinsx=35, showlegend=True,
                   legendgroup='1'), row=3, col=1)
    fig.add_trace(go.Histogram(x=df["NOX"], name="NOX", nbinsx=35, showlegend=True,
                   legendgroup='1'), row=3, col=2)
    fig.add_trace(go.Histogram(x=df["RM"], name="RM", nbinsx=35, showlegend=True,
                   legendgroup='1'), row=4, col=1)
    fig.add_trace(go.Histogram(x=df["MEDV"], name="MEDV", nbinsx=35, showlegend=True,
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
    fig.append_trace(go.Box(x=df["CRIM"], name="CRIM"), row=1, col=1)
    fig.append_trace(go.Box(x=df["ZN"],name="ZN"), row=2, col=1)
    fig.append_trace(go.Box(x=df["LSTAT"],name="LSTAT"), row=3, col=1)
    fig.append_trace(go.Box(x=df["INDUS"], name="INDUS"), row=4, col=1)

    # Column 2.
    fig.append_trace(go.Box(x=df["CHAS"], name="CHAS"), row=1, col=2)
    fig.append_trace(go.Box(x=df["NOX"],name="NOX"), row=2, col=2)
    fig.append_trace(go.Box(x=df["RM"],name="RM"), row=3, col=2)
    fig.append_trace(go.Box(x=df["MEDV"], name="MEDV"), row=4, col=2)

    # Editing the layout and showing the figure. 
    fig.update_layout(plot_bgcolor='#F8F8F6', title_text="Boxplot of Numerical Features", title_font_size=20, 
                title_font_family='Arial Black', title_x=0, title_y=0.98, height = 650, width = 950)
    fig.show()
    

 # ============================== 2_Base_Models ============================== #   


def kfold(scores=None):
    fig = go.Figure() # Creating blank figure. 

    # Adding indivdual boxplots to figure. 
    fig.add_trace(go.Box(y=list(scores[0]), name="LR"))
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


def scatterplot_pricecap(df1=None, df2=None):
    # Setting up our figure canvas. 
    fig = make_subplots(rows=1, cols=2,  subplot_titles=('With Price Cap', 'Without Price Cap'))

    fig.add_trace(go.Scatter(x=df1["LSTAT"], y=df1["MEDV"], mode="markers", name="Price Cap"), row=1, col=1, )     
    fig.add_trace(go.Scatter(x=df2["LSTAT"], y=df2["MEDV"],mode="markers", name="Without Price Cap"), 
                  row=1, col=2)      

    fig.update_layout(plot_bgcolor='#F8F8F6',height=450, width=950, title_text="Price Cap Analysis",
                          title_font_size=20, title_font_family='Arial Black')
    fig.update_yaxes(title='MEDV')
    fig.update_xaxes(title='LSTAT')
    fig.show() # Showing figure.
    
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


    fig.update_layout(height=550, width=900, title_text="Model Performance over 30 Iterations", plot_bgcolor='#F8F8F6', title_font_size=20, 
                       title_font_family='Arial Black')

    fig['layout']['yaxis']['title']=score_string
    fig['layout']['xaxis3']['title']='Iterations'

    fig.show()

    
def residual_plot(xaxis, yaxis):
    # PLotting the residual plot. 
    fig = px.scatter(x=xaxis, y=yaxis)
    fig.add_hline(y=0, line_width=2, line_color="black")  
    fig.update_layout(title='Residual Plot',
                           xaxis_title='Residuals',
                           yaxis_title='Actual',
                           height=400, width=650,  plot_bgcolor='#F8F8F6', title_font_size=20,  title_font_family='Arial Black')
    fig.show()
    
def distribution(xaxis=None):
    fig = px.histogram(x=xaxis)
    fig.update_layout(title='Distribution of Resiudal Error',
                           xaxis_title='Residuals',
                           yaxis_title='count',
                           height=400, width=650,  plot_bgcolor='#F8F8F6', title_font_size=20, title_font_family='Arial Black')
    fig.show()





    