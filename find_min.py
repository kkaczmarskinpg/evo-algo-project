import benchmark_functions as bf
from opfunu import cec_based

#benchmark_functions
#pip install benchmark_functions
# func = bf.f16_2024(n_dimensions=2)

# # print(func.suggested_bounds())
# print(func.minimum())

#print(func([2.202906, 1.570796, 1.284992, 1.923058, 1.72047, 1.570796, 1.454414, 1.756087, 1.655717, 1.570796]))

#cec
#pip install opfunu

func = cec_based.cec2014.F162014(ndim=30)
# print(func.bounds)
print(func.x_global)
print(func.f_global)


# definition = func.definition()
# print(definition)
# # print(func.suggested_bounds())
# minima = func.minima()
# print(minima)
# for x in minima:
#     print(x)

# func.show(showPoints=func.minima())
# func.show(asHeatMap=True)
#
