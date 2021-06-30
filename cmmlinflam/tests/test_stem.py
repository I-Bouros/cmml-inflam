#
# This file is part of CMMLINFLAM
# (https://github.com/I-Bouros/cmml-inflam.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#

import unittest

import cmmlinflam as ci


class TestStemGillespie(unittest.TestCase):
    """
    Test the 'StemGillespie' class.
    """
    def test__init__(self):
        algo = ci.StemGillespie()
        self.assertEqual(
            algo._output_names,
            ['WT', 'A', 'B'])
        self.assertEqual(
            algo._parameter_names,
            ['WT0', 'A0', 'B0', 'alpha', 's', 'r', 'mu_A', 'mu_B'])
        self.assertEqual(algo._n_outputs, 3)
        self.assertEqual(algo._n_parameters, 8)

    def test_n_outputs(self):
        algo = ci.StemGillespie()
        self.assertEqual(algo.n_outputs(), 3)

    def test_n_parameters(self):
        algo = ci.StemGillespie()
        self.assertEqual(algo.n_parameters(), 8)

    def test_output_names(self):
        algo = ci.StemGillespie()
        self.assertEqual(
            algo.output_names(),
            ['WT', 'A', 'B'])

    def test_parameter_names(self):
        algo = ci.StemGillespie()
        self.assertEqual(
            algo.parameter_names(),
            ['WT0', 'A0', 'B0', 'alpha', 's', 'r', 'mu_A', 'mu_B'])

    def test_set_outputs(self):
        algo = ci.StemGillespie()
        outputs = ['A', 'B']
        algo.set_outputs(outputs)

        with self.assertRaises(ValueError):
            outputs1 = ['W', 'A']
            algo.set_outputs(outputs1)

    def test_simulate_fixed_times(self):
        algo = ci.StemGillespie()
        parameters = [100, 0, 0, 0.5, 0.1, 0.1, 0.2, 0.3]

        output_algorithm = algo.simulate_fixed_times(parameters, 1, 30)

        self.assertEqual(
            output_algorithm.shape,
            (30, 3))

        with self.assertRaises(TypeError):
            parameters = [100, 0, 0, 0.5, 0.001, 0.01, 0.002, 0.003]
            algo.simulate_fixed_times(parameters, '1', 30)

        with self.assertRaises(ValueError):
            parameters = [100, 0, 0, 0.5, 0.001, 0.01, 0.002, 0.003]
            algo.simulate_fixed_times(parameters, 0, 30)

        with self.assertRaises(TypeError):
            parameters = [100, 0, 0, 0.5, 0.001, 0.01, 0.002, 0.003]
            algo.simulate_fixed_times(parameters, 1, 30.)

        with self.assertRaises(ValueError):
            parameters = [100, 0, 0, 0.5, 0.001, 0.01, 0.002, 0.003]
            algo.simulate_fixed_times(parameters, 1, -2)

        with self.assertRaises(ValueError):
            parameters = [100, 0, 0, 0.5, 0.001, 0.01, 0.002, 0.003]
            algo.simulate_fixed_times(parameters, 10, 3)

        with self.assertRaises(TypeError):
            parameters = (100, 0, 0, 0.5, 0.001, 0.01, 0.002, 0.003)
            algo.simulate_fixed_times(parameters, 1, 30)

        with self.assertRaises(ValueError):
            parameters = [100, 0, 0, 0.5, 0.001, 0.01, 0.002, 0.003, 0]
            algo.simulate_fixed_times(parameters, 1, 30)

        with self.assertRaises(TypeError):
            parameters = [100, '0', 0, 0.5, 0.001, 0.01, 0.002, 0.003]
            algo.simulate_fixed_times(parameters, 1, 30)

        with self.assertRaises(ValueError):
            parameters = [100, 0, -2, 0.5, 0.001, 0.01, 0.002, 0.003]
            algo.simulate_fixed_times(parameters, 1, 30)

        with self.assertRaises(TypeError):
            parameters = [100, 0, 0, '0.5', 0.001, 0.01, 0.002, 0.003]
            algo.simulate_fixed_times(parameters, 1, 30)

        with self.assertRaises(ValueError):
            parameters = [100, 0, 0, 0.5, -0.001, 0.01, 0.002, 0.003]
            algo.simulate_fixed_times(parameters, 1, 30)

        with self.assertRaises(TypeError):
            parameters = [100, 0, 0, 0.5, 0.001, 0.01, '0.002', 0.003]
            algo.simulate_fixed_times(parameters, 1, 30)

        with self.assertRaises(ValueError):
            parameters = [100, 0, 0, 0.5, 0.001, 0.01, 0.002, -0.003]
            algo.simulate_fixed_times(parameters, 1, 30)

    def test_simulate_criterion(self):
        algo = ci.StemGillespie()
        parameters = [100, 0, 0, 0.5, 0.1, 0.1, 0.2, 0.3]

        computation_time, final_state = algo.simulate_criterion(
            parameters, 0.2)

        self.assertEqual(
            final_state.shape,
            (3, ))

        self.assertEqual(type(computation_time), int)

        with self.assertRaises(TypeError):
            parameters = [100, 0, 0, 0.5, 0.001, 0.01, 0.002, 0.003]
            algo.simulate_criterion(parameters, '0.2')

        with self.assertRaises(ValueError):
            parameters = [100, 0, 0, 0.5, 0.001, 0.01, 0.002, 0.003]
            algo.simulate_criterion(parameters, -0.2)

        with self.assertRaises(ValueError):
            parameters = [100, 0, 0, 0.5, 0.001, 0.01, 0.002, 0.003]
            algo.simulate_criterion(parameters, 1.2)

        with self.assertRaises(TypeError):
            parameters = (100, 0, 0, 0.5, 0.001, 0.01, 0.002, 0.003)
            algo.simulate_criterion(parameters, 0.2)

        with self.assertRaises(ValueError):
            parameters = [100, 0, 0, 0.5, 0.001, 0.01, 0.002, 0.003, 0]
            algo.simulate_criterion(parameters, 0.2)

        with self.assertRaises(TypeError):
            parameters = [100, '0', 0, 0.5, 0.001, 0.01, 0.002, 0.003]
            algo.simulate_criterion(parameters, 0.2)

        with self.assertRaises(ValueError):
            parameters = [100, 0, -2, 0.5, 0.001, 0.01, 0.002, 0.003]
            algo.simulate_criterion(parameters, 0.2)

        with self.assertRaises(TypeError):
            parameters = [100, 0, 0, '0.5', 0.001, 0.01, 0.002, 0.003]
            algo.simulate_criterion(parameters, 0.2)

        with self.assertRaises(ValueError):
            parameters = [100, 0, 0, 0.5, -0.001, 0.01, 0.002, 0.003]
            algo.simulate_criterion(parameters, 0.2)

        with self.assertRaises(TypeError):
            parameters = [100, 0, 0, 0.5, 0.001, 0.01, '0.002', 0.003]
            algo.simulate_criterion(parameters, 0.2)

        with self.assertRaises(ValueError):
            parameters = [100, 0, 0, 0.5, 0.001, 0.01, 0.002, -0.003]
            algo.simulate_criterion(parameters, 0.2)


class TestStemGillespieTIMEVAR(unittest.TestCase):
    """
    Test the 'StemGillespieTIMEVAR' class.
    """
    def test__init__(self):
        algo = ci.StemGillespieTIMEVAR()
        self.assertEqual(
            algo._output_names,
            ['WT', 'A', 'B'])
        self.assertEqual(
            algo._parameter_names,
            ['WT0', 'A0', 'B0', 'alpha', 's', 'r', 'mu_A', 'mu_B'])
        self.assertEqual(algo._n_outputs, 3)
        self.assertEqual(algo._n_parameters, 8)

    def test_n_outputs(self):
        algo = ci.StemGillespieTIMEVAR()
        self.assertEqual(algo.n_outputs(), 3)

    def test_n_parameters(self):
        algo = ci.StemGillespieTIMEVAR()
        self.assertEqual(algo.n_parameters(), 8)

    def test_output_names(self):
        algo = ci.StemGillespieTIMEVAR()
        self.assertEqual(
            algo.output_names(),
            ['WT', 'A', 'B'])

    def test_parameter_names(self):
        algo = ci.StemGillespieTIMEVAR()
        self.assertEqual(
            algo.parameter_names(),
            ['WT0', 'A0', 'B0', 'alpha', 's', 'r', 'mu_A', 'mu_B'])

    def test_set_outputs(self):
        algo = ci.StemGillespieTIMEVAR()
        outputs = ['A', 'B']
        algo.set_outputs(outputs)

        with self.assertRaises(ValueError):
            outputs1 = ['W', 'A']
            algo.set_outputs(outputs1)

    def test_simulate_fixed_times(self):
        algo = ci.StemGillespieTIMEVAR()
        parameters = [100, 0, 0, 0.5, 0.1, 0.1, 0.2, 0.3]
        switch_times = [[0, 1], [5, 0], [10, 1], [20, 0]]

        output_algorithm = algo.simulate_fixed_times(
            parameters, switch_times, 1, 100)

        self.assertEqual(
            output_algorithm.shape,
            (100, 3))

        with self.assertRaises(TypeError):
            parameters = [100, 0, 0, 0.5, 0.001, 0.01, 0.002, 0.003]
            algo.simulate_fixed_times(parameters, switch_times, '1', 30)

        with self.assertRaises(ValueError):
            parameters = [100, 0, 0, 0.5, 0.001, 0.01, 0.002, 0.003]
            algo.simulate_fixed_times(parameters, switch_times, 0, 30)

        with self.assertRaises(TypeError):
            parameters = [100, 0, 0, 0.5, 0.001, 0.01, 0.002, 0.003]
            algo.simulate_fixed_times(parameters, switch_times, 1, 30.)

        with self.assertRaises(ValueError):
            parameters = [100, 0, 0, 0.5, 0.001, 0.01, 0.002, 0.003]
            algo.simulate_fixed_times(parameters, switch_times, 1, -2)

        with self.assertRaises(ValueError):
            parameters = [100, 0, 0, 0.5, 0.001, 0.01, 0.002, 0.003]
            algo.simulate_fixed_times(parameters, switch_times, 10, 3)

        with self.assertRaises(ValueError):
            switch_times = [0, 1, 2]
            algo.simulate_fixed_times(parameters, switch_times, 1, 30)

        with self.assertRaises(ValueError):
            switch_times = [[0], [1], [2]]
            algo.simulate_fixed_times(parameters, switch_times, 1, 30)

        with self.assertRaises(TypeError):
            switch_times = [[0, 1], ['5', 0], [10, 1], [20, 0]]
            algo.simulate_fixed_times(parameters, switch_times, 1, 30)

        with self.assertRaises(ValueError):
            switch_times = [[0, 1], [-5, 0], [10, 1], [20, 0]]
            algo.simulate_fixed_times(parameters, switch_times, 1, 30)

        with self.assertRaises(TypeError):
            switch_times = [[0, 1], [5, 0], [10, '1'], [20, 0]]
            algo.simulate_fixed_times(parameters, switch_times, 1, 30)

        with self.assertRaises(ValueError):
            switch_times = [[0, 1], [5, 0], [10, -1], [20, 0]]
            algo.simulate_fixed_times(parameters, switch_times, 1, 30)

        with self.assertRaises(ValueError):
            switch_times = [[2, 1], [5, 0], [10, 1], [20, 0]]
            algo.simulate_fixed_times(parameters, switch_times, 1, 30)

        with self.assertRaises(TypeError):
            parameters = (100, 0, 0, 0.5, 0.001, 0.01, 0.002, 0.003)
            algo.simulate_fixed_times(parameters, switch_times, 1, 30)

        with self.assertRaises(ValueError):
            parameters = [100, 0, 0, 0.5, 0.001, 0.01, 0.002, 0.003, 0]
            algo.simulate_fixed_times(parameters, switch_times, 1, 30)

        with self.assertRaises(TypeError):
            parameters = [100, '0', 0, 0.5, 0.001, 0.01, 0.002, 0.003]
            algo.simulate_fixed_times(parameters, switch_times, 1, 30)

        with self.assertRaises(ValueError):
            parameters = [100, 0, -2, 0.5, 0.001, 0.01, 0.002, 0.003]
            algo.simulate_fixed_times(parameters, switch_times, 1, 30)

        with self.assertRaises(TypeError):
            parameters = [100, 0, 0, '0.5', 0.001, 0.01, 0.002, 0.003]
            algo.simulate_fixed_times(parameters, switch_times, 1, 30)

        with self.assertRaises(ValueError):
            parameters = [100, 0, 0, 0.5, -0.001, 0.01, 0.002, 0.003]
            algo.simulate_fixed_times(parameters, switch_times, 1, 30)

        with self.assertRaises(TypeError):
            parameters = [100, 0, 0, 0.5, 0.001, 0.01, '0.002', 0.003]
            algo.simulate_fixed_times(parameters, switch_times, 1, 30)

        with self.assertRaises(ValueError):
            parameters = [100, 0, 0, 0.5, 0.001, 0.01, 0.002, -0.003]
            algo.simulate_fixed_times(parameters, switch_times, 1, 30)

    def test_simulate_criterion(self):
        algo = ci.StemGillespieTIMEVAR()
        parameters = [100, 0, 0, 0.5, 0.1, 0.1, 0.2, 0.3]
        switch_times = [[0, 1], [5, 0], [10, 1], [20, 0]]

        computation_time, final_state = algo.simulate_criterion(
            parameters, switch_times, 0.2)

        self.assertEqual(
            final_state.shape,
            (3, ))

        self.assertEqual(type(computation_time), int)

        with self.assertRaises(TypeError):
            parameters = [100, 0, 0, 0.5, 0.001, 0.01, 0.002, 0.003]
            algo.simulate_criterion(parameters, switch_times, '0.2')

        with self.assertRaises(ValueError):
            parameters = [100, 0, 0, 0.5, 0.001, 0.01, 0.002, 0.003]
            algo.simulate_criterion(parameters, switch_times, -0.2)

        with self.assertRaises(ValueError):
            parameters = [100, 0, 0, 0.5, 0.001, 0.01, 0.002, 0.003]
            algo.simulate_criterion(parameters, switch_times, 1.2)

        with self.assertRaises(ValueError):
            switch_times = [0, 1, 2]
            algo.simulate_criterion(parameters, switch_times, 0.2)

        with self.assertRaises(ValueError):
            switch_times = [[0], [1], [2]]
            algo.simulate_criterion(parameters, switch_times, 0.2)

        with self.assertRaises(TypeError):
            switch_times = [[0, 1], ['5', 0], [10, 1], [20, 0]]
            algo.simulate_criterion(parameters, switch_times, 0.2)

        with self.assertRaises(ValueError):
            switch_times = [[0, 1], [-5, 0], [10, 1], [20, 0]]
            algo.simulate_criterion(parameters, switch_times, 0.2)

        with self.assertRaises(TypeError):
            switch_times = [[0, 1], [5, 0], [10, '1'], [20, 0]]
            algo.simulate_criterion(parameters, switch_times, 0.2)

        with self.assertRaises(ValueError):
            switch_times = [[0, 1], [5, 0], [10, -1], [20, 0]]
            algo.simulate_criterion(parameters, switch_times, 0.2)

        with self.assertRaises(ValueError):
            switch_times = [[2, 1], [5, 0], [10, 1], [20, 0]]
            algo.simulate_criterion(parameters, switch_times, 0.2)

        with self.assertRaises(TypeError):
            parameters = (100, 0, 0, 0.5, 0.001, 0.01, 0.002, 0.003)
            algo.simulate_criterion(parameters, switch_times, 0.2)

        with self.assertRaises(ValueError):
            parameters = [100, 0, 0, 0.5, 0.001, 0.01, 0.002, 0.003, 0]
            algo.simulate_criterion(parameters, switch_times, 0.2)

        with self.assertRaises(TypeError):
            parameters = [100, '0', 0, 0.5, 0.001, 0.01, 0.002, 0.003]
            algo.simulate_criterion(parameters, switch_times, 0.2)

        with self.assertRaises(ValueError):
            parameters = [100, 0, -2, 0.5, 0.001, 0.01, 0.002, 0.003]
            algo.simulate_criterion(parameters, switch_times, 0.2)

        with self.assertRaises(TypeError):
            parameters = [100, 0, 0, '0.5', 0.001, 0.01, 0.002, 0.003]
            algo.simulate_criterion(parameters, switch_times, 0.2)

        with self.assertRaises(ValueError):
            parameters = [100, 0, 0, 0.5, -0.001, 0.01, 0.002, 0.003]
            algo.simulate_criterion(parameters, switch_times, 0.2)

        with self.assertRaises(TypeError):
            parameters = [100, 0, 0, 0.5, 0.001, 0.01, '0.002', 0.003]
            algo.simulate_criterion(parameters, switch_times, 0.2)

        with self.assertRaises(ValueError):
            parameters = [100, 0, 0, 0.5, 0.001, 0.01, 0.002, -0.003]
            algo.simulate_criterion(parameters, switch_times, 0.2)
