# Student Marks Predictor

A machine learning web application built with Streamlit that predicts student final examination marks based on their academic performance data.

## Features

- **Interactive UI**: Clean and user-friendly interface powered by Streamlit
- **ML Prediction**: Uses a trained machine learning model to predict final marks (G3)
- **Real-time Input**: Adjustable sliders for:
  - Study time per week (1-4 scale)
  - Number of classes missed (0-30)
  - First internal marks (G1, 0-20)
  - Second internal marks (G2, 0-20)
- **Instant Results**: Get predicted final marks on a scale of 0-20

## Tech Stack

- **Python 3.x**
- **Streamlit**: Web application framework
- **Scikit-learn**: Machine learning model
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Joblib**: Model serialization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/chinna464/student-marks-predictor.git
cd student-marks-predictor
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Project Structure

```
student-marks-predictor/
│
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── data/
│   └── student.csv          # Training dataset
├── src/
│   ├── train.py             # Model training script
│   ├── save_model.py        # Model saving utilities
│   └── predict.py           # Prediction functions
├── app.py                   # Main Streamlit application
├── model.pkl                # Trained ML model
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

## Model Training

To retrain the model with your own data:

1. Place your dataset in the `data/` directory
2. Run the training script:
```bash
python src/train.py
```

## How It Works

The application uses a regression model trained on student performance data. It takes four key features:
1. **Study Time**: Weekly study hours (1=Very Low, 2=Low, 3=Moderate, 4=High)
2. **Absences**: Number of classes missed
3. **G1**: First period grade (0-20 scale)
4. **G2**: Second period grade (0-20 scale)

Based on these inputs, it predicts **G3** (final grade on 0-20 scale).

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is open source and available under the MIT License.

## Author

Built with ❤️ using Python and Streamlit
