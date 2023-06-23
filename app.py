import streamlit as st
import pandas as pd
import joblib
import pickle 
import sklearn 
import datetime

vot_model = joblib.load("final.pkl")
xgboost_model = joblib.load("xgboost_pipeline.pkl")
catboost_model = joblib.load("catboost_pipeline.pkl")
Inputs = joblib.load("inputs.pkl")


def prediction(Airline, Source, Destination, Total_Stops, Additional_Info, Arrival_Date, Arrival_Time, Dep_Date, Dep_Time):
    Arrival_Datetime = datetime.datetime.combine(Arrival_Date, Arrival_Time)
    Dep_Datetime = datetime.datetime.combine(Dep_Date, Dep_Time)
    
    if Arrival_Datetime <= Dep_Datetime:
        st.error('Arrival date and time must be greater than Departure date and time')
    else:
        diff_time = (Arrival_Datetime - Dep_Datetime)
        test_df = pd.DataFrame(columns=Inputs)
        test_df.at[0,"Airline"] = Airline
        test_df.at[0,"Source"] = Source
        test_df.at[0,"Destination"] = Destination
        test_df.at[0,"Total_Stops"] = Total_Stops
        test_df.at[0,"Duration_Total_Minutes"] = diff_time.seconds/60
        test_df.at[0,"Additional_Info"] = Additional_Info
        test_df.at[0,"Arrival_Hour"] = Arrival_Time.hour
        test_df.at[0,"Arrival_Minute"] = Arrival_Time.minute
        test_df.at[0,"Dep_Hour"] = Dep_Time.hour
        test_df.at[0,"Dep_Minute"] = Dep_Time.minute
        test_df.at[0,"Dep_Month"] = Dep_Date.month
        test_df.at[0,"Dep_day"] = Dep_Date.day
        test_df.at[0,"Dep_WeekDay"] = Dep_Date.strftime("%A")[:3]

        st.dataframe(test_df)
        result = vot_model.predict(test_df)
        return result

    
def main():
    st.image('indian_airlines.jpg')
    Airline = st.selectbox("Airline" , ['Jet Airways', 'IndiGo', 'Air India',
                                        'Multiple carriers', 'SpiceJet', 'Vistara', 'Air Asia', 'GoAir'])
    Source = st.selectbox("Source" , ['Delhi', 'Kolkata', 'Banglore', 'Mumbai', 'Chennai'])
    Destination = st.selectbox("Destination" , ['Cochin', 'Banglore', 'Delhi', 'New Delhi', 'Hyderabad', 'Kolkata'])    
    
    dep_cols=st.columns(2)
    with dep_cols[0]:
        Dep_Date = st.date_input("Departure Date")
    with dep_cols[1]:
        Dep_Time = st.time_input("Departure Time")
        
    arr_cols=st.columns(2)
    with arr_cols[0]:
        Arrival_Date = st.date_input("Arrival Date")
    with arr_cols[1]:
        Arrival_Time = st.time_input("Arrival Time")
    
    Additional_Info = st.selectbox("Additional Info" , 
                                   ['No info', 'In-flight meal not included', 'No check-in baggage included', 'Other'])    
    Total_Stops = st.selectbox("Total Stops" , [0, 1, 2, 3, 4])

    if st.button("Predict", type='primary'):
        result = prediction(Airline, Source, Destination, Total_Stops, Additional_Info, Arrival_Date, Arrival_Time, Dep_Date, Dep_Time)
        if result:
            st.markdown(f'## Predicted Price will be : {round(result[0], 3)}')
        
if __name__ == '__main__':
    main()    
