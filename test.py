from flight_recording import (
    generate_and_save_flight_data,
    record_flight_for_seconds
)
from artificial_gps.model_tester import test_model_predictions
from artificial_gps.experiment1 import train_static_model
from artificial_gps.data import load_preprocessed_sequences

if __name__ == "__main__":
    # pass
    # # x, y = load_preprocessed_sequences()
    # # generate_and_save_flight_data()
    # record_flight_for_seconds(30)
    test_model_predictions("1st", "flight_2021:12:19_22:46:08_record_data.csv")
#

"""

position:
predicted offset: [-0.03224769  0.01676653 -0.01921006]
real offset:      [ 0.00613403 -0.00207138 -0.05758667]
time offset(seconds): [1.00502144]


position:
predicted offset: [ 1.5202974  -0.56063837  0.98933756]
real offset:      [ 1.53326416 -0.53096771  0.88059235]
time offset(seconds): [2.00404275]


position:
predicted offset: [ 7.9477315 -3.0826027  3.0867174]
real offset:      [ 7.67837524 -2.81509781  2.86894989]
time offset(seconds): [3.00306406]


position:
predicted offset: [18.897411  -7.5809536  4.509599 ]
real offset:      [18.39425659 -7.10147095  4.16374969]
time offset(seconds): [4.00208563]


position:
predicted offset: [ 33.572487 -13.774695   3.873484]
real offset:      [ 32.64427185 -13.08042908   3.42718506]
time offset(seconds): [5.00110669]


position:
predicted offset: [135.61829   -51.21158     1.2472289]
real offset:      [132.64064026 -49.83224487   0.44979095]
time offset(seconds): [10.00221363]


position:
predicted offset: [243.87286  -88.43443    4.882187]
real offset:      [238.7136116  -86.40119934   3.71697998]
time offset(seconds): [15.00332006]


position:
predicted offset: [ 352.06094  -125.52974     8.546377]
real offset:      [ 344.76079178 -122.87741852    7.01797485]
time offset(seconds): [20.00142669]



Process finished with exit code 0


"""