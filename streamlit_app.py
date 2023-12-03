import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
@st.cache_data
def load_model():
    model = joblib.load('./classifier.joblib')
    return model

model = load_model()

# Define the data preprocessing function
def preprocess_input(df):
    # Convert Date and Time columns to datetime
    df['Data/Fecha/Date'] = pd.to_datetime(df['Data/Fecha/Date'])
    df['Hora/Hora/Time'] = pd.to_datetime(df['Hora/Hora/Time'])

    # Extract Month, Day, and Hour
    df['Month'] = df['Data/Fecha/Date'].dt.month
    df['Day'] = df['Data/Fecha/Date'].dt.day
    df['Hour'] = df['Hora/Hora/Time'].dt.hour

    # Pivot and prepare the data
    hourly_data = df.pivot_table(index='ID', columns='Hour', values='Consum (L/h)/Consumo (L/h)/Consumption (L/h)', aggfunc='mean')
    all_hours = range(24)
    hourly_data = hourly_data.reindex(columns=all_hours, fill_value=0)
    hourly_data.fillna(0, inplace=True)

    return df, hourly_data



# Define the leak detector
def detect_leaks(original_data, processed_hourly_data, window_size=3):


     # Ensure the 'Hour' column in original_data is an integer
    original_data['Hour'] = original_data['Hour'].astype(int)
    
    # If 'ID' is a column in processed_hourly_data, drop it to get only the hourly consumption columns
    if 'ID' in processed_hourly_data.columns:
        hourly_averages = processed_hourly_data.drop(columns=['ID']).iloc[0].astype(float)
    else:
        # If 'ID' is the index, just take the first row's values as the hourly averages
        hourly_averages = processed_hourly_data.iloc[0].astype(float)

    # Compare hourly consumption against hourly averages
    original_data['Above_Average'] = original_data.apply(
        lambda row: row['Consum (L/h)/Consumo (L/h)/Consumption (L/h)'] > hourly_averages[row['Hour']], axis=1
    )

    # Detect a potential leak as sustained above-average consumption
    original_data['Consecutive'] = (original_data['Above_Average'] & (original_data['Hour'].diff().fillna(1) == 1)).cumsum()

    # Flag potential leaks where there are at least `window_size` consecutive hours of above-average consumption
    leak_groups = original_data[original_data['Above_Average']].groupby('Consecutive').filter(lambda x: len(x) >= window_size)

    # Extract the potential leak events

    
    potential_leaks = leak_groups['Data/Fecha/Date'].dt.date.unique()
    
    return potential_leaks

# Usage in Streamlit
# Assuming original_data and processed_hourly_data are already defined and preprocessed...



# Streamlit application layout
st.title('Aïgues de Barcelona Analysis App')

# File uploader
uploaded_file = st.file_uploader(label="Puja el teu consum d'aïgua", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    original_data, processed_hourly_data = preprocess_input(data)
    
     # Preparing to plot the pattern
    pattern_plot = processed_hourly_data.transpose()
    pattern_plot.columns = ['Consumption']
    pattern_plot.index.name = 'Hour'
    pattern_plot.reset_index(inplace=True)
    st.header('Patró de consum horari')
    st.area_chart(pattern_plot.set_index('Hour'))

    if data['Ús/Uso/Use'].iloc[0] == 'Domèstic/Doméstico/Domestic':

        # Run model prediction
        predictions = model.predict(processed_hourly_data)
        st.write("This house belongs to the category", predictions[0])


    # Preparing to plot the original data
    original_data['Data/Fecha/Date'] = pd.to_datetime(original_data['Data/Fecha/Date'])
    original_data['Hora/Hora/Time'] = pd.to_datetime(original_data['Hora/Hora/Time'])
    original_data['Date'] = original_data['Data/Fecha/Date'].dt.date
    original_data['Hour'] = original_data['Hora/Hora/Time'].dt.hour

    # Reshape the data
    reshaped_data = original_data[['Date', 'Hour', 'Consum (L/h)/Consumo (L/h)/Consumption (L/h)']]
    reshaped_data['Datetime'] = pd.to_datetime(reshaped_data['Date'].astype(str) + ' ' + reshaped_data['Hour'].astype(str) + ':00:00')
    reshaped_data['Anomaly']='None'

    reshaped_data_non_zero = reshaped_data[reshaped_data['Consum (L/h)/Consumo (L/h)/Consumption (L/h)'] > 0].copy()

    # Calculate IQR and determine outliers
    Q1 = reshaped_data_non_zero['Consum (L/h)/Consumo (L/h)/Consumption (L/h)'].quantile(0.15)
    Q3 = reshaped_data_non_zero['Consum (L/h)/Consumo (L/h)/Consumption (L/h)'].quantile(0.85)
    IQR = Q3 - Q1
    threshold = 3 * IQR
    reshaped_data_non_zero.loc[
    (reshaped_data_non_zero['Consum (L/h)/Consumo (L/h)/Consumption (L/h)'] < (Q1 - threshold)) |
    (reshaped_data_non_zero['Consum (L/h)/Consumo (L/h)/Consumption (L/h)'] > (Q3 + threshold)),
    'Anomaly'] = 'Overuse'
    

    # Plotting with Streamlit
    st.header("Consum horari d'aigua amb indicació d'anomalies")
    st.scatter_chart(
    reshaped_data_non_zero.set_index('Date'),
    y='Consum (L/h)/Consumo (L/h)/Consumption (L/h)',
    color='Anomaly',  # This column will determine the color of each point
    size='Consum (L/h)/Consumo (L/h)/Consumption (L/h)'  # Optional, if you want to vary the size as well
)
    
    potential_leaks = detect_leaks(original_data, processed_hourly_data, window_size=3)
    
    if potential_leaks.size > 0:
        st.subheader("Em trobat potencials fugues els dies:")
        for date in potential_leaks:
            st.write(f"- {date.strftime('%Y-%m-%d')}")
    else:
        st.write("No em trobat cap consum sospitos de ser una fuga")


   



