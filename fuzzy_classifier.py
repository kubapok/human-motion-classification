import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyClassifier(object):

    def __init__(self):
        #  Define result dictionary
        self.MOTION_DICT = {1: 'going left', 2: 'going right', 3: 'standing', 4: 'sitting', 5: 'lying'}

        #  Define input variables
        self.v_move = ctrl.Antecedent(np.arange(-300000, 300001, 1), 'v_move')
        self.h_move = ctrl.Antecedent(np.arange(-300000, 300001, 1), 'h_move')
        self.height = ctrl.Antecedent(np.arange(0, 501, 1), 'height')
        self.width = ctrl.Antecedent(np.arange(0, 501, 1), 'width')

        #  Define output variable
        self.motion = ctrl.Consequent(np.arange(0, 7, 1), 'motion')

        #  Fuzzify input variables
        self.v_move['down'] = fuzz.trimf(self.v_move.universe, [-300000, -300000, 0])
        self.v_move['center'] = fuzz.trimf(self.v_move.universe, [-300000, 0, 300000])
        self.v_move['up'] = fuzz.trimf(self.v_move.universe, [0, 300000, 300000])

        self.h_move['left'] = fuzz.trimf(self.h_move.universe, [-300000, -300000, 0])
        self.h_move['center'] = fuzz.trimf(self.h_move.universe, [-300000, 0, 300000])
        self.h_move['right'] = fuzz.trimf(self.h_move.universe, [0, 300000, 300000])

        self.height['low'] = fuzz.trimf(self.height.universe, [0, 0, 250])
        self.height['medium'] = fuzz.trimf(self.height.universe, [0, 250, 500])
        self.height['high'] = fuzz.trimf(self.height.universe, [250, 500, 500])

        self.width['narrow'] = fuzz.trimf(self.width.universe, [0, 0, 250])
        self.width['medium'] = fuzz.trimf(self.width.universe, [0, 250, 500])
        self.width['wide'] = fuzz.trimf(self.width.universe, [250, 500, 500])

        #  Fuzzify output variable
        self.motion['going left'] = fuzz.trimf(self.motion.universe, [0, 1, 2])
        self.motion['going right'] = fuzz.trimf(self.motion.universe, [1, 2, 3])
        self.motion['standing'] = fuzz.trimf(self.motion.universe, [2, 3, 4])
        self.motion['sitting'] = fuzz.trimf(self.motion.universe, [3, 4, 5])
        self.motion['lying'] = fuzz.trimf(self.motion.universe, [4, 5, 6])

        #  Define rules
        self.rule1 = ctrl.Rule(self.h_move['left'] & self.v_move['center'] & self.height['high'] & self.width['medium'], self.motion['going left'])
        self.rule2 = ctrl.Rule(self.h_move['right'] & self.v_move['center'] & self.height['high'] & self.width['medium'], self.motion['going right'])
        self.rule3 = ctrl.Rule(self.h_move['center'] & self.v_move['center'] & self.height['high'] & self.width['narrow'], self.motion['standing'])
        self.rule4 = ctrl.Rule(self.h_move['center'] & self.v_move['down'] & self.height['medium'] & self.width['narrow'], self.motion['sitting'])
        self.rule5 = ctrl.Rule(self.h_move['center'] & self.v_move['down'] & self.height['low'] & self.width['wide'], self.motion['lying'])

        self.rules = [self.rule1, self.rule2, self.rule3, self.rule4, self.rule5]       

        #  Create control system
        self.controler = ctrl.ControlSystem(self.rules)

        #  Create system simulator
        self.simulator = ctrl.ControlSystemSimulation(self.controler)


    def plot_variables(self):
        #self.v_move.view()
        #self.h_move.view()
        self.height.view()
        #self.width.view()
        #self.motion.view()
        input("Press Enter to continue...")


    def plot_result(self):
        self.motion.view(sim=self.simulator)
        input("Press Enter to continue...")


    def classify(self, data):
        #  Pass inputs to the simulation
        try:
            self.simulator.input['v_move'] = data[0]
            self.simulator.input['h_move'] = data[1]
            self.simulator.input['height'] = data[2]
            self.simulator.input['width'] = data[3]
        except:
            return ''

        #  Crunch the numbers
        try:
            self.simulator.compute()
        except:
            return ''

        #  Return the result
        #self.plot_result()
        try:
            result = int(round(self.simulator.output['motion']))
        except:
            return ''
        return self.MOTION_DICT[result]
