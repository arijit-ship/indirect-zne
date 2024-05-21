from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class ZeroNoiseExtrapolation:
    def __init__(self, data_points):
        """
        Initialize with a list of data points.
        
        Each data point should be a list where the first three elements are 
        the noise parameters and the last element is the energy value.
        """
        self.data_points = data_points

        self.noise_param = [point[:3] for point in data_points]
        self.energies = [point[3] for point in data_points]
        

    def linear_extrapolation(self):
        """
        Fit a linear regression model using the noise parameters and energy values.
        """
        self.model = LinearRegression()
        self.model.fit(self.noise_param, self.energies)
        return self.model.predict([[0, 0, 0]])[0] # type: ignore
    
def polynomial_extrapolation(self, degree: int):
    """
    Fit a linear regression model using the noise parameters and energy values.
    """
   
    # Generate polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(self.noise_param)

    # Fit linear regression model on polynomial features
    model = LinearRegression()
    model.fit(X_poly, energies)

    # Define the point (0,0,0) for prediction
    point_to_predict = np.array([[0, 0, 0]])

    # Transform the point into polynomial features
    point_to_predict_poly = poly.transform(point_to_predict)

    # Predict the energy at the point
    predicted_energy = model.predict(point_to_predict_poly)
        

# Example usage
data_points = [
    [0.1, 0.2, 0.3, 1.0],
    [0.2, 0.1, 0.4, 1.1],
    [0.3, 0.2, 0.2, 1.2],
    # Add more data points as needed
]

zne = ZeroNoiseExtrapolation(data_points)
extrapolated_value = zne.linear_extrapolation()

print("Extrapolated value:", extrapolated_value)
