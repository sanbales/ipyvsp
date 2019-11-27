import numpy as np
import traitlets as trt


class ParsecAirfoil(trt.HasTraits):
    """
    A traitleted PARametric SECtion (PARSEC) Airfoil.
    
    Credit:
    * Based on Marc Bodmer's implementation (https://github.com/mbodmer/libairfoil/blob/master/libairfoil/parsec.py)
    
    References:
    - "Parametric Airfoils and Wings" by Helmut Sobieczky (http://www.as.dlr.de/hs/h-pdf/H141.pdf)
    
    TODO:
    * Add blending
    * Make thickness and camber the source parameters for `upper_z` and `lower_z`?
    
    """
    
    upper_x = trt.Float( 0.400, min=0.01, max=1.0, help="Upper crest location horizontal coordinate")
    upper_z = trt.Float( 0.075, min=-1.0, max=1.0, help="Upper crest location vertical coordinate")
    upper_c = trt.Float(-0.100, min=-1.0, max=1.0, help="Upper crest location curvature")
    
    lower_x = trt.Float( 0.400, min=0.01, max=1.0, help="Lower crest location horizontal coordinate")
    lower_z = trt.Float(-0.075, min=-1.0, max=1.0, help="Lower crest location vertical coordinate")
    lower_c = trt.Float( 0.100, min=-1.0, max=1.0, help="Lower crest location curvature")
    
    le_radius = trt.Float(0.01, min=0.0, max=1.0, help="Leading edge radius")
    te_z = trt.Float(0.0, min=0.0, max=1.0, help="Trailing edge vertical coordinate")
    te_alpha = trt.Float(0.0, min=-np.pi, max=np.pi, help="Trailing edge direction angle")
    te_beta = trt.Float(np.radians(20.0), min=-np.pi, max=np.pi, help="Trailing edge wedge angle")

    te_thickness = trt.Float(0.0, min=0.0, max=1.0, help="Trailing edge thickness, max=1.0 equates to a thickness of 1% chord")
    # blending = trt.Float(0.0, min=-1.0, max=1.0, help="Blend with NACA0012 (-1) or Whitcomb (+1) airfoils")
    
    _upper_coefficients = trt.Tuple(help="Coefficients used to calculate the upper surface")
    _lower_coefficients = trt.Tuple(help="Coefficients used to calculate the lower surface")
    
    num_points = trt.Int(200, min=50, max=1000, help="The number of horizontally equally distributed points to use")
    coordinates = trt.Tuple(help="The (x,y) points that make up the airfoil, defined clockwise starting at the leading edge")
    
    @trt.default("_upper_coefficients")
    def _default_upper_coefficients(self):
        return self._calculate_coefficients(upper=True)
    
    @trt.default("_lower_coefficients")
    def _default_lower_coefficients(self):
        return self._calculate_coefficients(upper=False)
    
    @trt.default("coordinates")
    def _default_coordinates(self):
        return self._calculate_coordinates()
    
    @trt.observe("upper_x", "upper_z", "upper_c", "le_radius", "te_z", "te_alpha", "te_beta", "te_thickness")  # "blending"
    def _update_upper_coefficients(self, *_):
        self._upper_coefficients = self._calculate_coefficients(upper=True)
        
    @trt.observe("lower_x", "lower_z", "lower_c", "le_radius", "te_z", "te_alpha", "te_beta", "te_thickness")  # "blending"
    def _update_lower_coefficients(self, *_):
        self._lower_coefficients = self._calculate_coefficients(upper=False)
    
    @trt.observe("num_points", "_upper_coefficients", "_lower_coefficients")
    def _update_coordinates(self, *_):
        self.coordinates = self._calculate_coordinates()
    
    @staticmethod
    def _calculate_surface_coordinates(coefficients, x):
        return list(
            zip(
                x, 
                sum(
                    k * x ** (0.5 + i)
                    for k, i in zip(coefficients, range(len(coefficients)))
                )
            )
        )
    
    def _calculate_coordinates(self):
        x = 0.5 * (1 + np.cos(np.linspace(np.pi, 0, self.num_points)))
        return (
            self._calculate_surface_coordinates(self._upper_coefficients, x) +
            list(reversed(self._calculate_surface_coordinates(self._lower_coefficients, x)))
        )
    
    def _calculate_coefficients(self, upper: bool):
        return tuple(np.linalg.solve(self.A(upper), self.B(upper)))
    
    def A(self, upper: bool) -> np.ndarray:
        """Calculate the A matrix."""
        x = self.upper_x if upper else self.lower_x
        
        return np.array([
            [1.0,           1.0,          1.0,         1.0,          1.0,          1.0        ],
            [x**0.5,        x**1.5,       x**2.5,      x**3.5,       x**4.5,       x**5.5     ],
            [0.5,           1.5,          2.5,         3.5,          4.5,          5.5        ],
            [0.5*x**-0.5,   1.5*x**0.5,   2.5*x**1.5,  3.5*x**2.5,   4.5*x**3.5,   5.5*x**4.5 ],
            [-0.25*x**-1.5, 0.75*x**-0.5, 3.75*x**0.5, 8.75*x**1.5, 15.75*x**2.5, 24.75*x**3.5],
            [1.0,           0.0,          0.0,         0.0,          0.0,          0.0        ],
        ])

    def B(self, upper: bool) -> np.ndarray:
        """Calculate the B vector."""
        sign = 1 if upper else -1
        
        return np.array([
            self.te_z + 0.005 * sign * self.te_thickness,
            self.upper_z if upper else self.lower_z,
            np.tan(self.te_alpha - sign * 0.5 * self.te_beta),
            0.0,
            self.upper_c if upper else self.lower_c,
            sign * np.sqrt(2 * self.le_radius),
        ])
