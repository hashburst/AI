from flask import Flask, render_template, jsonify
import plotly.graph_objs as go
import pandas as pd
import os

app = Flask(__name__)

# Path dove si trovano i dati del mining
DATA_PATH = "/path/to/data/mining_data.csv"

# Funzione per leggere i dati delle performance del miner
def get_mining_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        return df
    else:
        # Simulazione dati se il file non esiste
        data = {
            "time": pd.date_range(start="2023-09-01", periods=100, freq='H'),
            "accepted_share_ratio": np.random.uniform(0.90, 1.0, size=100),
            "rejected_share_ratio": np.random.uniform(0.0, 0.1, size=100),
            "profit": np.random.uniform(0.1, 1.0, size=100)
        }
        df = pd.DataFrame(data)
        return df

# Endpoint per restituire i dati in formato JSON
@app.route("/data")
def data():
    df = get_mining_data()
    return df.to_json(orient='records')

# Homepage della dashboard
@app.route("/")
def index():
    return render_template("index.html")

# Codice per i grafici Plotly
def generate_plot():
    df = get_mining_data()

    trace1 = go.Scatter(
        x=df['time'],
        y=df['accepted_share_ratio'],
        mode='lines',
        name='Accepted Share Ratio'
    )

    trace2 = go.Scatter(
        x=df['time'],
        y=df['rejected_share_ratio'],
        mode='lines',
        name='Rejected Share Ratio'
    )

    trace3 = go.Scatter(
        x=df['time'],
        y=df['profit'],
        mode='lines',
        name='Profit'
    )

    layout = go.Layout(
        title='Miner Performance Over Time',
                                      xaxis={'title': 'Time'},
                                      yaxis={'title': 'Metric'},
                                      hovermode='closest'
                                  )
                              
                                  fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
                                  return fig.to_html()
                              
                              @app.route("/plot")
                              def plot():
                                  plot_html = generate_plot()
                                  return plot_html
                              
                              if __name__ == "__main__":
                                  app.run(debug=True)
