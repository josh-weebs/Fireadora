import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib
import matplotlib.pyplot as plt
import pickle
model=pickle.load(open('model.pkl','rb'))

matplotlib.use('Agg')
from PIL import Image

st.title('Forest Fire Prediction/Analysis')
image=Image.open('forest.jpg')
st.image(image,use_column_width=True)
@st.cache
def predict_forest(oxygen,humidity,temperature):
    input=np.array([[oxygen,humidity,temperature]]).astype(np.float64)
    prediction=model.predict_proba(input)
    pred='{0:.{1}f}'.format(prediction[0][0], 2)
    return float(pred)

def main():
    activities=['EDA','Visualisation','Prediction','Effects']
    option=st.sidebar.selectbox('Selection option:',activities)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    if option=='EDA':
        st.subheader("Exploratory data analysis")
        data=st.file_uploader("Upload your dataset:",type=['csv','xlsx','txt','json'])
        data=pd.read_csv("cov1.csv")
        if data is not None:
            st.success("Data successfully uploaded")
            df=pd.read_csv("cov1.csv")
            st.dataframe(df.head(50))

            if st.checkbox("Display shape"):
                st.write(df.shape)
            if st.checkbox("Display columns"):
                st.write(df.columns)
            if st.checkbox("Select multiple columns"):
                selected_column=st.multiselect('Select prefered columns:',df.columns)
                df1=df[selected_column]
                st.dataframe(df1)  
            if st.checkbox("Display summary"):
                st.write(df1.describe().T) 
            if st.checkbox("Display datatypes"):
                st.write(df.dtypes)
            if st.checkbox("Display Correlation of data various columns"):
                st.write(df.corr())
            
        
    




    elif option=='Visualisation':
        st.subheader("Visualisation")
        # data=st.file_uploader("Upload your dataset:",type=['csv','xlsx','txt','json'])
        data=pd.read_csv("cov1.csv")
        if data is not None:
            st.success("Data successfully uploaded")
            df=pd.read_csv("cov1.csv")
            st.dataframe(df.head(50))

            if st.checkbox('Select Multiple Columns to plot'):
                selected_column=st.multiselect('Select your preffered columns',df.columns)
                selected_column.append("Cover_Type")
                df1=df[selected_column]
                # st.dataframe(df1)

            if st.checkbox('Display Heatmap'):
                st.write(sb.heatmap(df1.corr(),vmax=1,vmin=0, xticklabels=True, yticklabels=True,square=True,annot=True,cmap='viridis'))
                st.pyplot()
            if st.checkbox('Display Pairplot'):
                dataf=pd.read_csv("names.csv")
                st.dataframe(dataf)
                # st.write(df.Cover_Type.value_counts())
                st.write(sb.pairplot(df1,hue="Cover_Type",palette="husl"))
                st.pyplot()
            if st.checkbox('Display Countplot'):
                # st.write(df.Cover_Type.value_counts())
                st.write(sb.countplot(x='Cover_Type',data=df))
                st.pyplot()
            if st.checkbox("Display Wilderness Density"):
                trees = pd.read_csv("cov1.csv")
                trees['Wilderness_Area_Type'] = (trees.iloc[:, 11:15] == 1).idxmax(1)
                wilderness_areas = sorted(trees['Wilderness_Area_Type'].value_counts().index.tolist())
                for area in wilderness_areas:
                    subset = trees[trees['Wilderness_Area_Type'] == area]
                    sb.kdeplot(subset["Cover_Type"], label=area, linewidth=1)   
                plt.ylabel("Density")
                plt.xlabel("Cover_Type")
                plt.legend()
                plt.title("Density of Cover Types Among Different Wilderness Areas", size=14)
                st.pyplot()

            if st.checkbox("boxplot"):
                st.write("#### Select column to visualize: ")
                columns = df.columns.tolist()
                class_name = columns[-1]
                column_name = st.selectbox("",columns)
                st.write(sb.boxplot(x=class_name, y=column_name, palette="husl", data=df))
                st.pyplot()

                   


                        

    if option=='Prediction':
        st.subheader("Predection")
        html_temp = """
        <div style="background-color:#025246 ;padding:10px">
        <h2 style="color:white;text-align:center;">Forest Fire Prediction ML App </h2>
        </div>
        """
        st.markdown(html_temp, unsafe_allow_html=True)

        oxygen = st.text_input("Oxygen","")
        humidity = st.text_input("Humidity","")
        temperature = st.text_input("Temperature","")
        safe_html="""  
        <div style="background-color:#F4D03F;padding:10px >
        <h2 style="color:white;text-align:center;"> Your forest is safe</h2>
        </div>
        """
        danger_html="""  
        <div style="background-color:#F08080;padding:10px >
        <h2 style="color:black ;text-align:center;"> Your forest is in danger</h2>
        </div>
        """

        if st.button("Predict"):
            output=predict_forest(oxygen,humidity,temperature)
            st.success('The probability of fire taking place is {}'.format(output))

            if output > 0.5:
                st.markdown(danger_html,unsafe_allow_html=True)
            else:
                st.markdown(safe_html,unsafe_allow_html=True)

    elif option=='Effects':
        st.subheader("Effects due to forest fire")
        dataf=pd.read_csv("forest_fire.csv")
        dataf.sort_values("Fire Occurrence", inplace=True)
        filter=dataf["Fire Occurrence"]==1
        dataf.where(filter,inplace=True)
        dataf=dataf.dropna()
        st.dataframe(dataf.head(100))
        st.write("### Regions where Forest Fire Occured")
        st.map(dataf)

        st.write("### Pollution due to Forest Fire")
        df = pd.read_csv('pollution.csv')
      
        df= df.rename(columns = {" pm25": "pm25", 
                         " pm10":"pm10", 
                         " o3": "o3",
                         ' no2' : 'no2',
                         ' so2' : 'so2',
                         ' co' : 'co'})

        
        df['date'] = pd.to_datetime(df.date)
        df21 = df.loc[df['date'] > '2019-7-01']
        df21 = df21.sort_values(by = 'date')
        df21.drop(13, inplace=True)
        df21.replace(' ', '0', inplace=True)

        dates = df21['date']
        pm25 = df21['pm25']
        pm25 = [int(i) for i in pm25]
        o3 = df21['o3']
        o3 = [int(i) for i in o3]
        no2 = df21['no2']
        no2 = [int(i) for i in no2]
        so2 = df21['so2']
        so2 = [int(i) for i in so2]


        plt.figure(figsize=(10,8))
        ploti = st.selectbox("", ["pm25","O3","NO2","SO2"])
        if ploti=="pm25" :
            plt.plot(dates,pm25)
            plt.ylabel('PM25')
            plt.xlabel('MONTHLY ACTIVITY')
        elif ploti=="O3":
            plt.plot(dates,o3)
            plt.ylabel('O3')
            plt.xlabel('MONTHLY ACTIVITY')
        elif ploti=="NO2":
            plt.plot(dates,no2)
            plt.ylabel('NO2')
            plt.xlabel('MONTHLY ACTIVITY')
        if ploti=="SO2":
           plt.plot(dates,so2)
           plt.ylabel('SO2') 
           plt.xlabel('MONTHLY ACTIVITY')

        st.pyplot()

        st.write("### Losses due to Forest Fire")
        df21 = pd.read_csv('losses.csv')


        year = df21['Year']
        Fires = df21['Fires']
        Fires = [int(i) for i in Fires]
        Firedeath = df21['FireDeaths']
        Firedeath = [int(i) for i in Firedeath]
        FireInjuries = df21['FireInjuries']
        FireInjuries = [int(i) for i in FireInjuries]
        Loss = df21['ActualFireDollarLoss']
        Loss = [int(i) for i in Loss] 


        plt.figure(figsize=(10,8))
        ploti = st.selectbox("", ["Death","Injured","Money loss"])
        if ploti=="Death" :
            plt.plot(year,Firedeath)
            plt.xlabel('Year')
            plt.ylabel('Deaths')
        elif ploti=="Injured" :
            plt.plot(year,FireInjuries)
            plt.xlabel('Year')
            plt.ylabel('injured')
        elif ploti=="Money loss" :
            plt.plot(year,Loss)
            plt.xlabel('Year')
            plt.ylabel('Loss (in Billion)')


        st.pyplot()


if __name__=='__main__':
    main()