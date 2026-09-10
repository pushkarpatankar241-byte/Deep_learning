# Assignment 4 - Time-Series Stock Price Forecasting using Stacked LSTM

> **BRACT's Vishwakarma Institute of Technology, Pune**  
> Department of Computer Science & Engineering (Artificial Intelligence)  
> **Practical No. 4 | Deep Learning (TY SEM-1, 2026-27)**  
> **Student**: Pushkar Patankar | **PRN**: 12410410 | **Roll No.**: 63 | **Division E, Batch 3**  

---

## 1. Problem Statement & Objective

Financial time-series data exhibit complex non-linear temporal dynamics, non-stationarity, volatility clustering, and long-range dependencies. Traditional statistical models (such as ARIMA or Holt-Winters) often struggle to capture deep non-linear patterns across multi-day lookback windows.

The primary objective of this assignment is to design, implement, evaluate, and deploy a **Deep Stacked Long Short-Term Memory (LSTM)** recurrent neural network for multi-scale financial time-series forecasting using historical stock price data (Tata Global Beverages / Stock Data). The model forecasts future closing prices and provides 30-day autoregressive future projections with 95% confidence intervals without lookahead bias or data leakage.

---

## 2. Theoretical Foundations: Why LSTM for Time-Series?

Standard Recurrent Neural Networks (RNNs) suffer from the **Vanishing and Exploding Gradient Problem** over long sequences due to continuous matrix multiplications during Backpropagation Through Time (BPTT). LSTMs resolve this by maintaining an explicit internal **Cell State ($\mathbf{C}_t$)** regulated by three continuous non-linear gating mechanisms:

1. **Forget Gate ($\mathbf{f}_t$)**: Determines which past information to discard from the cell state:
   $$\mathbf{f}_t = \sigma\left(\mathbf{W}_f \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f\right)$$

2. **Input Gate ($\mathbf{i}_t$) & Candidate Memory ($\mathbf{\tilde{C}}_t$)**: Selects and scales new information for state memory update:
   $$\mathbf{i}_t = \sigma\left(\mathbf{W}_i \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i\right)$$
   $$\mathbf{\tilde{C}}_t = \tanh\left(\mathbf{W}_c \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c\right)$$

3. **Cell State Update ($\mathbf{C}_t$)**: Linear combination of retained past memory and modulated candidate memory:
   $$\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \mathbf{\tilde{C}}_t$$

4. **Output Gate ($\mathbf{o}_t$) & Hidden State ($\mathbf{h}_t$)**: Determines the filtered external activation emitted for time-step $t$:
   $$\mathbf{o}_t = \sigma\left(\mathbf{W}_o \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o\right)$$
   $$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t)$$

---

## 3. End-to-End Pipeline & Methodology

```
[Raw Historical Stock CSV]
           │
           ▼
1. Preprocessing & Chronological Sorting  ──► Monotonic date indexing (prevent lookahead bias)
           │
           ▼
2. Exploratory Data Analysis (EDA)       ──► Trend lines, 50/200-day SMAs, Daily Return volatility & Cross-Correlation
           │
           ▼
3. Normalization (MinMaxScaler)          ──► Scale target Close price to [0, 1] range for gradient stability
           │
           ▼
4. Sliding Window Sequence Formulation    ──► Lookback window = 60 days (X: [t-59, ..., t] ──► y: [t+1])
           │
           ▼
5. Chronological Train-Test Split         ──► Strict 80% Train / 20% Test partition (Zero temporal leakage)
           │
           ▼
6. Deep Stacked LSTM Architecture         ──► LSTM(64) + Dropout(0.2) + LSTM(64) + Dropout(0.2) + Dense(32) + Dense(1)
           │
           ▼
7. Optimization & Training Execution      ──► Adam (lr=0.001) + MSE Loss + ModelCheckpoint + EarlyStopping
           │
           ▼
8. Post-Evaluation & Metrics (Test Set)   ──► Inverse scale predictions; compute MSE, RMSE, MAE, MAPE, R², Directional Acc
           │
           ▼
9. Autoregressive 30-Day Future Forecast  ──► Multi-step horizon projection with dynamic 95% Confidence Interval Fan Bands
```

---

## 4. Model Architecture & Hyperparameters

### Network Specification
- **Input Dimension**: $(60, 1)$ (60 consecutive business days of historical normalized close prices)
- **Layer 1**: `LSTM(units=64, return_sequences=True)` + `Dropout(rate=0.2)`
- **Layer 2**: `LSTM(units=64, return_sequences=False)` + `Dropout(rate=0.2)`
- **Layer 3**: `Dense(units=32, activation='relu')`
- **Output Layer**: `Dense(units=1, activation='linear')`

### Training Parameters
| Hyperparameter | Value / Configuration |
|:---|:---|
| **Lookback Window** | 60 Trading Days |
| **Optimizer** | Adam ($\eta = 0.001$) |
| **Loss Function** | Mean Squared Error (MSE) |
| **Batch Size** | 32 |
| **Epochs** | 25 (with EarlyStopping & Checkpoint) |
| **Train / Test Split** | 80% / 20% (Strict Chronological) |

---

## 5. Quantitative Results & Evaluation Metrics

The model was evaluated on the unseen 20% test partition after inverse transforming the predictions back to the original Indian Rupee (INR) scale:

| Metric | Measured Score | Interpretation |
|:---|:---:|:---|
| **Coefficient of Determination ($R^2$)** | **0.9616** | **96.16%** of price variance explained by the model |
| **Root Mean Squared Error (RMSE)** | **10.62 INR** | Low deviation from true ground truth stock prices |
| **Mean Absolute Error (MAE)** | **7.87 INR** | Average absolute pricing error $< 8$ INR |
| **Mean Absolute Percentage Error (MAPE)** | **3.39 %** | High commercial-grade prediction precision ($< 5\%$) |
| **Mean Squared Error (MSE)** | **112.79** | Minimal variance in squared residual errors |
| **Directional Trend Accuracy** | **53.55 %** | Positive trend-following capability on daily movements |

---

## 6. Autoregressive 30-Day Future Forecast

Using the last 60 consecutive trading days from the dataset, the trained Stacked LSTM was run in an autoregressive feedback loop to generate the next **30 business days** future price path with expanding 95% confidence intervals:

| Day | Forecast Date | Expected Close Price (INR) | 95% Lower Bound | 95% Upper Bound |
|:---:|:---:|:---:|:---:|:---:|
| **1** | 2018-10-01 | **231.14** | 221.83 | 240.45 |
| **2** | 2018-10-02 | **231.45** | 218.28 | 244.61 |
| **3** | 2018-10-03 | **231.58** | 215.46 | 247.71 |
| **5** | 2018-10-05 | **231.53** | 210.71 | 252.35 |
| **10** | 2018-10-12 | **230.66** | 201.22 | 260.10 |
| **15** | 2018-10-19 | **229.59** | 193.54 | 265.64 |
| **30** | 2018-11-09 | **227.18** | 181.42 | 272.94 |

---

## 7. Key Findings & Discussion

1. **High Explanatory Power ($R^2 = 0.9616$)**: The 2-layer stacked LSTM effectively extracts both short-term momentum and medium-term cyclical signals from the 60-day window.
2. **Low Relative Error ($\text{MAPE} = 3.39\%$)**: Demonstrates that normalizing via `MinMaxScaler` and applying intermediate Dropout ($0.2$) effectively prevented over-reacting to transient noise.
3. **Multi-Step Horizon Behavior**: Autoregressive forecasting exhibits smooth mean-reversion behavior with expanding uncertainty bands, reflecting realistic financial market dynamics.

---

## 8. Repository Structure

```
Assignment_4_LSTM_Stock_Prediction/
├── DL_Assignment_4_Pushkar_Patankar.ipynb  # Complete pre-executed notebook with EDA, LSTM, Evaluation & Forecasting
└── README.md                              # Detailed experimental report & mathematical formulation
```
