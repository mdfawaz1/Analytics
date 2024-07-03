from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.decomposition import PCA
import openai
import concurrent.futures

app = Flask(__name__)
openai.api_key = 'sk-bQCApnOIPq41D0td3EOopSur71xsYReAvWd2iAWe'

def load_data_source(file_path):
    return pd.read_csv(file_path)

def clean_data(df):
    return df.dropna()

# Feature engineering focused only on numeric columns
def engineer_features(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    col_pairs = [(col1, col2) for col1 in numeric_columns for col2 in numeric_columns if col1 != col2]
    
    engineered_features = pd.DataFrame(index=df.index)
    for col1, col2 in col_pairs:
        engineered_features[f'{col1}_x_{col2}'] = df[col1] * df[col2]
    
    return pd.concat([df, engineered_features], axis=1)

# Correlation analysis focused only on numeric columns
def find_correlations(df):
    correlations = []
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    for col in numeric_columns:
        for other_col in numeric_columns:
            if col != other_col:
                valid_data = df[[col, other_col]].dropna()

                if len(valid_data) > 0:
                    X = sm.add_constant(valid_data[other_col])
                    y = valid_data[col]

                    try:
                        model = sm.OLS(y, X).fit()
                        if model.pvalues[other_col] < 0.05:
                            correlations.append({
                                'target': col,
                                'feature': other_col,
                                'coefficient': model.params[other_col],
                                'p_value': model.pvalues[other_col]
                            })
                    except Exception as e:
                        print(f"Error processing correlation between {col} and {other_col}: {e}")

    return correlations

def get_interpretation(corr):
    target = corr['target']
    feature = corr['feature']
    coefficient = corr.get('coefficient', None)
    p_value = corr['p_value']
    
    messages = [
        {
            "role": "system",
            "content": "You are a knowledgeable data analyst providing insights into the correlation between variables. Provide a short and powerful one-line insight without explaining the underlying variables, just suggest the best possible interpretation of the data."
        },
        {
            "role": "user",
            "content": f"""
                Given the following correlation information:
                - Target: {target}
                - Feature: {feature}
                - Coefficient: {coefficient}
                - p-value: {p_value}

                Provide an interpretation of the correlation.
            """
        }
    ]
    
    try:
        print(f"Sending request to OpenAI for {target} and {feature}...")
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1000
        )
        print(f"Received response from OpenAI for {target} and {feature}.")
        corr['interpretation'] = response.choices[0].message['content'].strip()
    except Exception as e:
        print(f"Error getting interpretation for {target} and {feature}: {e}")
        corr['interpretation'] = f"Error: {e}"
    
    return corr

def find_hidden_patterns(df):
    numeric_df = df.select_dtypes(include=[np.number])
    pca = PCA(n_components=min(len(numeric_df.columns), len(numeric_df)))
    principal_components = pca.fit_transform(numeric_df)
    explained_variance = pca.explained_variance_ratio_
    
    patterns = []
    for i, variance in enumerate(explained_variance):
        patterns.append({
            'component': i + 1,
            'explained_variance': variance
        })
    
    return patterns

def generate_summary(correlations):
    summary = "Summary of significant correlations:\n"
    for corr in correlations[:10]:
        summary += f"- {corr['interpretation']}\n"
    return summary.strip()

@app.route('/analyze', methods=['POST'])
def analyze_endpoint():
    user_input = request.json
    file_path = user_input.get('file_path')
    analysis_type = user_input.get('analysis_type', 'correlations')

    if not file_path:
        return jsonify({'error': 'File path is required'}), 400

    try:
        data = load_data_source(file_path)
        print(f"Loaded data from {file_path}:\n{data.head()}")
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    print(f"Data:\n{data.head()}")
    cleaned_data = clean_data(data)
    print(f"Cleaned data:\n{cleaned_data.head()}")

    feature_engineered_data = engineer_features(cleaned_data)
    print(f"Feature engineered data:\n{feature_engineered_data.head()}")

    if analysis_type == 'correlations':
        correlations = find_correlations(feature_engineered_data)
        print(f"Found {len(correlations)} correlations.")
        
        interpretations = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_corr = {executor.submit(get_interpretation, corr): corr for corr in correlations}
            
            for future in concurrent.futures.as_completed(future_to_corr):
                corr = future_to_corr[future]
                try:
                    interpretations.append(future.result(timeout=30))
                except Exception as e:
                    print(f"Error processing correlation {corr['target']} and {corr['feature']}: {e}")
        
        result = interpretations
        summary = generate_summary(interpretations)
        result.append({"summary": summary})
    elif analysis_type == 'hidden_patterns':
        result = find_hidden_patterns(feature_engineered_data)
    else:
        return jsonify({'error': f'Unknown analysis type: {analysis_type}'}), 400

    return jsonify(result)

@app.route('/notificationping', methods=['POST'])
def notification_ping():
    return jsonify({'message': 'Notification ping received'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
