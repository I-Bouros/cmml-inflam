#
# StemGillespie Class
#
# This file is part of CMMLINFLAM
# (https://github.com/I-Bouros/cmml-inflam.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#
"""
This script contains code for the forward simulation of the STEM cells
population, both mutated variants and wild types, involved in hematopoesis
in a given tumor.

It uses a Gillespie algorithm with fixed time step of 1.

"""

import numpy as np
from scipy.stats import uniform


class StemGillespie(object):
    r"""StemGillespie Class:

    Base class for the forward simulation of the evolution of a population
    of STEM cells.

    Three types of cells are considered: those which present mustations that
    give selectional advantage irrespective of environmental conditions (e.g.
    KRAS) (A), those which present mustations that give selectional advantage
    dependent on environmental conditions (e.g. increasing TNF or IGN) (B)
    and the wild type cells (WT).

    Cells decay at the same rate independent of their type and devide with
    rates which illsutate their selectional advantage. A wild type cell (WT)
    can mutate to a cell of type A, respectively a cell of type B with constant
    given rates of mutation.

    The system of equations that describe the different phenomena that can
    occur:

    ..math:
        :nowrap:

            \begin{eqnarray}
                WT \xrightarrow{m} \emptyset
                A \xrightarrow{m} \emptyset
                B \xrightarrow{m} \emptyset
                \emptyset \xrightarrow{\alpha_{WT}} WT
                \emptyset \xrightarrow{\alpha_{A}} A
                \emptyset \xrightarrow{\alpha_{B}} B
                WT \xrightarrow{\mu_{A}} A
                WT \xrightarrow{\mu_{B}} B
            \end{eqnarray}

    where m is the rate of decay, :math:`\alpha_{WT}`, :math:`\alpha_{A}`,
    and :math:`\alpha_{B}` are the growth rates for the WT, A and B cell
    type respectively and :math:`\mu_{A}` and :math:`\mu_{B}` are the rate
    of mutation of a WT cell into A cell and respectively, a B cell type.

    The total cell population is considered constant so the division of a cell
    is always followed by the death of a cell and vice versa.

    """
    def __init__(self):
        super(StemGillespie, self).__init__()

        # Assign default values
        self._output_names = ['WT', 'A', 'B']
        self._parameter_names = [
            'WT0', 'A0', 'B0', 'alpha', 's', 'r', 'mu_A', 'mu_B']

        # The default number of outputs is 3,
        # i.e. WT, A and B
        self._n_outputs = len(self._output_names)
        # The default number of outputs is 8,
        # i.e. 3 initial conditions and 5 parameters
        self._n_parameters = len(self._parameter_names)

        self._output_indices = np.arange(self._n_outputs)

    def n_outputs(self):
        """
        Returns the number of outputs.

        """
        return self._n_outputs

    def n_parameters(self):
        """
        Returns the number of parameters.

        """
        return self._n_parameters

    def output_names(self):
        """
        Returns the (selected) output names.

        """
        names = [self._output_names[x] for x in self._output_indices]
        return names

    def parameter_names(self):
        """
        Returns the parameter names.

        """
        return self._parameter_names

    def set_outputs(self, outputs):
        """
        Checks existence of outputs.

        """
        for output in outputs:
            if output not in self._output_names:
                raise ValueError(
                    'The output names specified must be in correct forms')

        output_indices = []
        for output_id, output in enumerate(self._output_names):
            if output in outputs:
                output_indices.append(output_id)

        # Remember outputs
        self._output_indices = output_indices
        self._n_outputs = len(outputs)

    def _prob_WT_to_B(self, i_WT, i_A, i_B):
        """
        Computes the probability of losing a WT cell and gaining a B cell
        when there is a change in the counts of the
        different species of cell in the tumor.

        This event can occur either through the mutation of a WT to a B,
        or the simultanious division of a B cell which kills a WT cell.

        Parameters
        ----------
        i_WT
            (int) number of wildtype cells (WT) in the tumor at current time
            point.
        i_A
            (int) number of 1st type mutated cells (A) in the tumor at current
            time point.
        i_B
            (int) number of 2nd type mutated cells (B) in the tumor at current
            time point.

        """
        mu = self.mu_A + self.mu_B

        # Compute probability of change through division
        prob_WT_die = i_WT/self.N
        tot_growth_rate = self.alpha_WT * i_WT + (
            self.alpha_A * i_A) + self.alpha_B * i_B
        prob_B_divide = (self.alpha_B * i_B) / tot_growth_rate
        divis = (1-mu) * prob_WT_die * prob_B_divide

        # Compute probability of change through mutation
        prob_WT_divide = (self.alpha_WT * i_WT) / tot_growth_rate
        mutat = self.mu_B * prob_WT_die * prob_WT_divide

        return (divis + mutat)

    def _prob_WT_to_A(self, i_WT, i_A, i_B):
        """
        Computes the probability of losing a WT cell and gaining a A cell
        when there is a change in the counts of the
        different species of cell in the tumor.

        This event can occur either through the mutation of a WT to a A,
        or the simultanious division of a A cell which kills a WT cell.

        Parameters
        ----------
        i_WT
            (int) number of wildtype cells (WT) in the tumor at current time
            point.
        i_A
            (int) number of 1st type mutated cells (A) in the tumor at current
            time point.
        i_B
            (int) number of 2nd type mutated cells (B) in the tumor at current
            time point.

        """
        mu = self.mu_A + self.mu_B

        # Compute probability of change through division
        prob_WT_die = i_WT/self.N
        tot_growth_rate = self.alpha_WT * i_WT + (
            self.alpha_A * i_A) + self.alpha_B * i_B
        prob_A_divide = (self.alpha_A * i_A) / tot_growth_rate
        divis = (1-mu) * prob_WT_die * prob_A_divide

        # Compute probability of change through mutation
        prob_WT_divide = (self.alpha_WT * i_WT) / tot_growth_rate
        mutat = self.mu_A * prob_WT_die * prob_WT_divide

        return (divis + mutat)

    def _prob_B_to_WT(self, i_WT, i_A, i_B):
        """
        Computes the probability of losing a B cell and gaining a WT cell
        when there is a change in the counts of the
        different species of cell in the tumor.

        This event can only occur either through the simultanious division of
        a WT cell which kills a B cell.

        Parameters
        ----------
        i_WT
            (int) number of wildtype cells (WT) in the tumor at current time
            point.
        i_A
            (int) number of 1st type mutated cells (A) in the tumor at current
            time point.
        i_B
            (int) number of 2nd type mutated cells (B) in the tumor at current
            time point.

        """
        mu = self.mu_A + self.mu_B

        # Compute probability of change through division
        prob_B_die = i_B/self.N
        tot_growth_rate = self.alpha_WT * i_WT + (
            self.alpha_A * i_A) + self.alpha_B * i_B
        prob_WT_divide = (self.alpha_WT * i_WT) / tot_growth_rate
        divis = (1-mu) * prob_B_die * prob_WT_divide

        return divis

    def _prob_A_to_WT(self, i_WT, i_A, i_B):
        """
        Computes the probability of losing a A cell and gaining a WT cell
        when there is a change in the counts of the
        different species of cell in the tumor.

        This event can only occur either through the simultanious division of
        a WT cell which kills a A cell.

        Parameters
        ----------
        i_WT
            (int) number of wildtype cells (WT) in the tumor at current time
            point.
        i_A
            (int) number of 1st type mutated cells (A) in the tumor at current
            time point.
        i_B
            (int) number of 2nd type mutated cells (B) in the tumor at current
            time point.

        """
        mu = self.mu_A + self.mu_B

        # Compute probability of change through division
        prob_A_die = i_A/self.N
        tot_growth_rate = self.alpha_WT * i_WT + (
            self.alpha_A * i_A) + self.alpha_B * i_B
        prob_WT_divide = (self.alpha_WT * i_WT) / tot_growth_rate
        divis = (1-mu) * prob_A_die * prob_WT_divide

        return divis

    def one_step_gillespie(self, i_WT, i_A, i_B):
        """
        Computes one step in the Gillespie algorithm to determine the
        counts of the different species of cells living in the tumor at
        present.

        Parameters
        ----------
        i_WT
            (int) number of wildtype cells (WT) in the tumor at current time
            point.
        i_A
            (int) number of 1st type mutated cells (A) in the tumor at current
            time point.
        i_B
            (int) number of 2nd type mutated cells (B) in the tumor at current
            time point.

        """
        # Generate random number
        u = uniform.rvs(size=1)

        # Compute propensities
        propens_1 = self._prob_WT_to_B(i_WT, i_A, i_B)
        propens_2 = self._prob_WT_to_A(i_WT, i_A, i_B)
        propens_3 = self._prob_B_to_WT(i_WT, i_A, i_B)
        propens_4 = self._prob_A_to_WT(i_WT, i_A, i_B)

        propens = np.array([propens_1, propens_2, propens_3, propens_4])
        sum_propens = np.empty(propens.shape)
        for e in range(propens.shape[0]):
            sum_propens[e] = np.sum(propens[:(e+1)])

        # Total propensity
        tot_propens = propens_1 + propens_2 + propens_3 + propens_4

        frac_propens = (1/tot_propens) * sum_propens

        if u < frac_propens[0]:
            i_WT += -1
            i_B += 1
        elif (u >= frac_propens[0]) and (u < frac_propens[1]):
            i_WT += -1
            i_A += 1
        elif (u >= frac_propens[1]) and (u < frac_propens[2]):
            i_WT += 1
            i_B += -1
        else:
            i_WT += 1
            i_A += -1

        return (i_WT, i_A, i_B)

    def gillespie_algorithm(self, times):
        """
        Runs the Gillespie algorithm for the STEM cell population
        for the given times.

        Parameters
        ----------
        times
            (list) Vector of the times for which the run the Gillespie
            algorithm.

        """
        # Split compartments into their types
        i_WT, i_A, i_B = self.init_cond

        solution = np.empty((len(times), 3), dtype=np.int)
        for t, _ in enumerate(times):
            solution[t, :] = [i_WT, i_A, i_B]
            i_WT, i_A, i_B = self.one_step_gillespie(i_WT, i_A, i_B)

        return({'state': solution})

    def simulate(self, parameters, start_time, end_time):
        r"""
        Computes the number of each type of cell in a given tumor between the
        given time points.

        Parameters
        ----------
        parameters
            (list) List of quantities that characterise the STEM cells cycle in
            this order: the initial counts for each type of cell (i_WT, i_A,
            i_B), the growth rate for the WT, the boosts in selection given to
            the mutated A and B variant respectively and the mutation rates
            with which a WT cell transforms into an A and B variant,
            respectively.
        start_time
            (int) Time from which we start the simulation of the tumor.
        end_time
            (int) Time at which we end the simulation of the tumor.

        """
        # Check correct format of output
        if not isinstance(start_time, int):
            raise TypeError('Start time of siumlation must be integer.')
        if start_time <= 0:
            raise ValueError('Start time of siumlation must be > 0.')

        if not isinstance(end_time, int):
            raise TypeError('End time of siumlation must be integer.')
        if end_time <= 0:
            raise ValueError('Start time of siumlation must be > 0.')

        if start_time > end_time:
            raise ValueError('End time must be after start time.')

        if not isinstance(parameters, list):
            raise TypeError('Parameters must be given in a list format.')
        if len(parameters) != 8:
            raise ValueError('List of parameters needs to be of length 8.')
        for _ in range(3):
            if not isinstance(parameters[_], int):
                raise TypeError(
                    'Initial cell count must be integer.')
            if parameters[_] < 0:
                raise ValueError('Initial cell count must be => 0.')
        for _ in range(3, 6):
            if not isinstance(parameters[_], (float, int)):
                raise TypeError(
                    'Growth rate must be integer or float.')
            if parameters[_] < 0:
                raise ValueError('Growth rate must be => 0.')
        for _ in range(6, 8):
            if not isinstance(parameters[_], (float, int)):
                raise TypeError(
                    'Mutation rate must be integer or float.')
            if parameters[_] < 0:
                raise ValueError('Mutation rate must be => 0.')

        # Split parameters into the features of the model
        # initial conditions
        self.init_cond = parameters[:3]
        self.N = sum(self.init_cond)

        # growth rates
        alpha, s, r = parameters[3:6]
        self.alpha_WT = alpha
        self.alpha_A = alpha + s
        self.alpha_B = alpha + r

        # mutation rates
        self.mu_A = parameters[6]
        self.mu_B = parameters[7]

        times = range(start_time, end_time+1)
        sol = self.gillespie_algorithm(times)

        output = sol['state']

        return output
