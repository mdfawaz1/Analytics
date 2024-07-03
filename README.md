app.py:
Its for getting the correlations after combining the data from two sources based on the merge field.
It gives the individual correlations along with a summary of correlations. 
The command to test it:

curl -X POST http://localhost:5001/analyze -H "Content-Type: application/json" -d '{
  "query": "find all the correlations in my data sources",
  "data_sources": ["customer_purchases", "customer_demographics"],
  "analysis_type": "correlations",            
  "target_columns": ["purchase_amount", "age", "income"]
}'

prediction.py:
It is for giving the prediction of the future outcomes depending on the historical data.
It is done by taking the average of the days values on an hourly basis and predicts the future outcomes
The command to run it:

streamlit run prediction.py
and load the building_data.csv file through the streamlit UI 

correlation.py:
It works on a single datasource and completely operates on the numeric data.
The command to test it:

curl -X POST http://localhost:5001/analyze \ 
-H "Content-Type: application/json" \                     
-d '{                                                             
    "file_path": "combined_customer_data.csv",
    "analysis_type": "correlations"                     
}'
