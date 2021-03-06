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

    Three types of cells are considered - those which present mutations that
    give selectional advantage irrespective of environmental conditions (A),
    those which present mutations that give selectional advantage dependent
    on environmental conditions (B) and the wild type cells (WT).

    Cells decay at the same rate independent of their type and devide with
    rates which illsutate their selectional advantage. A wild type cell (WT)
    can mutate to a cell of type A, respectively a cell of type B with constant
    given rates of mutation.

    The system of equations that describe the isolated possible events that can
    occur

    .. math::
        :nowrap:

        \begin{eqnarray}
            WT &\xrightarrow{m} \emptyset \\
            A  &\xrightarrow{m} \emptyset \\
            B  &\xrightarrow{m} \emptyset \\
            \emptyset  &\xrightarrow{\alpha_{WT}} WT \\
            \emptyset  &\xrightarrow{\alpha_{A}} A \\
            \emptyset  &\xrightarrow{\alpha_{B}} B \\
            WT  &\xrightarrow{\mu_{A}} A \\
            WT  &\xrightarrow{\mu_{B}} B
        \end{eqnarray}

    where m is the rate of decay, :math:`\alpha_{WT}`, :math:`\alpha_{A}`,
    and :math:`\alpha_{B}` are the growth rates for the WT, A and B cell
    type respectively and :math:`\mu_{A}` and :math:`\mu_{B}` are the rate
    of mutation of a WT cell into A cell and respectively, a B cell type.
    For this class we consider the temporal selectional advatange of the
    B cells always present.

    The total cell population is considered constant so the division of a cell
    is always simultaneous to the death of a cell.

    Therefore, the actual system of equations that describes the model is

    .. math::
        :nowrap:

        \begin{eqnarray}
            WT + WT &\xrightarrow{P_{WT \rightarrow A}} WT + A \\
            WT + WT &\xrightarrow{P_{WT \rightarrow B}} WT + B \\
            A + WT &\xrightarrow{P_{A \rightarrow B}} B + WT \\
            A + WT &\xrightarrow{P_{A \rightarrow WT}} WT + WT \\
            B + WT &\xrightarrow{P_{B \rightarrow A}} A + WT \\
            A + WT &\xrightarrow{P_{WT \rightarrow B}} A + B \\
            B + WT &\xrightarrow{P_{B \rightarrow WT}} WT + WT \\
            B + WT &\xrightarrow{P_{WT \rightarrow A}} A + B \\
            A + B &\xrightarrow{P_{A \rightarrow WT}} B + WT \\
            A + B &\xrightarrow{P_{B \rightarrow WT}} A + WT
        \end{eqnarray}

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
        or the simultaneous division of a B cell which kills a WT cell.

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
        or the simultaneous division of a A cell which kills a WT cell.

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

        This event can only occur either through the simultaneous division of
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

        This event can only occur either through the simultaneous division of
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

    def _prob_A_to_B(self, i_WT, i_A, i_B):
        """
        Computes the probability of losing a A cell and gaining a B cell
        when there is a change in the counts of the
        different species of cell in the tumor.

        This event can only occur either through the simultaneous division of
        a B cell which kills a A cell.

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
        prob_B_divide = (self.alpha_B * i_B) / tot_growth_rate
        divis = (1-mu) * prob_A_die * prob_B_divide

        return divis

    def _prob_B_to_A(self, i_WT, i_A, i_B):
        """
        Computes the probability of losing a B cell and gaining a A cell
        when there is a change in the counts of the
        different species of cell in the tumor.

        This event can only occur either through the simultaneous division of
        a A cell which kills a B cell.

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
        prob_A_divide = (self.alpha_A * i_A) / tot_growth_rate
        divis = (1-mu) * prob_B_die * prob_A_divide

        return divis

    def one_step_gillespie(self, i_WT, i_A, i_B):
        """
        Computes one step in the Gillespie algorithm to determine the
        counts of the different species of cells living in the tumor at
        present. Returns time to next reaction and the tuple state of the
        system.

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
        # Generate random number for reaction and time to next reaction
        u, u1 = uniform.rvs(size=2)

        # Time to next reaction
        tau = np.log(1/u1)

        # Compute propensities
        propens_1 = self._prob_WT_to_B(i_WT, i_A, i_B)
        propens_2 = self._prob_WT_to_A(i_WT, i_A, i_B)
        propens_3 = self._prob_B_to_WT(i_WT, i_A, i_B)
        propens_4 = self._prob_A_to_WT(i_WT, i_A, i_B)
        propens_5 = self._prob_A_to_B(i_WT, i_A, i_B)
        propens_6 = self._prob_B_to_A(i_WT, i_A, i_B)

        propens = np.array([
            propens_1, propens_2, propens_3, propens_4,
            propens_5, propens_6])
        sum_propens = np.empty(propens.shape)
        for e in range(propens.shape[0]):
            sum_propens[e] = np.sum(propens[:(e+1)]) / np.sum(propens)

        if u < sum_propens[0]:
            i_WT += -1
            i_B += 1
        elif (u >= sum_propens[0]) and (u < sum_propens[1]):
            i_WT += -1
            i_A += 1
        elif (u >= sum_propens[1]) and (u < sum_propens[2]):
            i_WT += 1
            i_B += -1
        elif (u >= sum_propens[2]) and (u < sum_propens[3]):
            i_WT += 1
            i_A += -1
        elif (u >= sum_propens[3]) and (u < sum_propens[4]):
            i_B += 1
            i_A += -1
        elif (u >= sum_propens[4]) and (u < sum_propens[5]):
            i_A += 1
            i_B += -1

        return (tau, i_WT, i_A, i_B)

    def gillespie_algorithm_fixed_times(self, start_time, end_time):
        """
        Runs the Gillespie algorithm for the STEM cell population
        for the given times.

        Parameters
        ----------
        start_time
            (int) Time from which we start the simulation of the tumor.
        end_time
            (int) Time at which we end the simulation of the tumor.

        """
        # Create timeline vector
        times = np.arange(start_time, end_time+0.5, 1, dtype=np.int)
        interval = end_time - start_time + 1

        # Split compartments into their types
        i_WT, i_A, i_B = self.init_cond

        large_solution = []
        time_solution = []
        solution = np.empty((interval, 3), dtype=np.int)
        current_time = start_time
        while current_time <= end_time:
            time_solution.append(current_time)
            large_solution.append([i_WT, i_A, i_B])
            tau, i_WT, i_A, i_B = self.one_step_gillespie(i_WT, i_A, i_B)
            current_time += tau

        eval_indices = np.where(
            np.array([(t in time_solution) for t in times]))[0].tolist()

        ind_in_times = []
        j = 0
        for i, t in enumerate(eval_indices):
            if t < eval_indices[-1]:
                ind_in_times.extend([j]*(eval_indices[i+1]-eval_indices[i]))
            else:
                ind_in_times.extend([j]*(times[-1]-eval_indices[i]))
            j += 1

        for t in range(interval):
            solution[t, :] = large_solution[ind_in_times[t]]

        return({'state': solution})

    def simulate_fixed_times(self, parameters, start_time, end_time):
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
        self._check_times(start_time, end_time)

        self._check_parameters_format(parameters)
        self._set_parameters(parameters)

        sol = self.gillespie_algorithm_fixed_times(start_time, end_time)

        output = sol['state']

        return output

    def _check_times(self, start_time, end_time):
        """
        Checks format of start and end of simulation window.

        """
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

    def _set_parameters(self, parameters):
        """
        Split parameters into the features of the model.

        """
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

    def gillespie_algorithm_criterion(self, criterion):
        """
        Runs the Gillespie algorithm for the STEM cell population
        until a criterion is met.

        Parameters
        ----------
        criterion
            (list of 2 lists) List of percentage thresholds of cell types in
            the population for disease to be triggered and another containing
            the type of threshold imposed.

        """
        # Split compartments into their types
        i_WT, i_A, i_B = self.init_cond

        time_to_criterion = 0
        while all(self._evaluate_criterion(criterion, i_WT, i_A, i_B)):
            tau, i_WT, i_A, i_B = self.one_step_gillespie(i_WT, i_A, i_B)
            time_to_criterion += tau

        return ({
            'time': int(time_to_criterion),
            'state': np.array([i_WT, i_A, i_B], dtype=np.int)})

    def _evaluate_criterion(self, criterion, i_WT, i_A, i_B):
        """
        Evaluates if criterion for stopping the simulation is met.

        """
        state = [i_WT, i_A, i_B]
        state_criterion = []
        for _, c in enumerate(criterion[0]):
            if c is not None:
                if criterion[1][_] == 'less':
                    state_criterion.append(state[_] < c * self.N)
                elif criterion[1][_] == 'more':
                    state_criterion.append(state[_] > c * self.N)
        return state_criterion

    def simulate_criterion(self, parameters, criterion):
        r"""
        Computes the number of each type of cell in a given tumor until a
        criterion is met.

        Parameters
        ----------
        parameters
            (list) List of quantities that characterise the STEM cells cycle in
            this order: the initial counts for each type of cell (i_WT, i_A,
            i_B), the growth rate for the WT, the boosts in selection given to
            the mutated A and B variant respectively and the mutation rates
            with which a WT cell transforms into an A and B variant,
            respectively.
        criterion
            (list of 2 lists) List of percentage thresholds of cell types in
            the population for disease to be triggered and another containing
            the type of threshold imposed.

        """
        # Check correct format of output
        self._check_criterion(criterion)

        self._check_parameters_format(parameters)
        self._set_parameters(parameters)

        sol = self.gillespie_algorithm_criterion(criterion)

        computation_time = sol['time']
        final_state = sol['state']

        return computation_time, final_state

    def _check_criterion(self, criterion):
        """
        Checks format of the criterion input for the simulation.

        """
        if not isinstance(criterion, list):
            raise TypeError(
                'Simulation criterion storage format must be a list.')
        if len(criterion) != 2:
            raise ValueError(
                'Simulation criterion storage format must be a list \
                    containing two distinct lists.')

        for part in criterion:
            if not isinstance(part, list):
                raise TypeError(
                    'Each part in the simulation criterion storage format \
                        must be a list.')
            if len(part) != 3:
                raise ValueError(
                    'Simulation criterion storage format must be a list of \
                        length 3.')

        for _, c in enumerate(criterion[0]):
            if not isinstance(c, (float, int)) and (c is not None):
                raise TypeError(
                    'Threshold value for siumlation must be integer, float or \
                        None.')
            if c is not None:
                if c < 0:
                    raise ValueError('Threshold value for siumlation must \
                        be >= 0.')
                if c > 1:
                    raise ValueError('Threshold value for siumlation must \
                        be <= 1.')
                if criterion[1][_] not in ['less', 'more']:
                    raise ValueError('For no given threshold value, we must\
                        have either the `less` or `more` keyword.')
            else:
                if criterion[1][_] is not None:
                    raise ValueError('For no given threshold value, we must\
                        have no keyword.')

        if criterion[0] == [None, None, None]:
            raise ValueError('Cannot have all criterion thresholds absent.')

        if sum(filter(None, criterion[0])) > 1:
            raise ValueError('Cannot have all criterion thresholds sum to more \
                than 1.')

    def _check_parameters_format(self, parameters):
        """
        Checks the format of the `paramaters` input in the simulation methods.

        """
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

    def gillespie_algorithm_fixation(self):
        """
        Runs the Gillespie algorithm for the STEM cell population
        until fixation.

        """
        # Split compartments into their types
        i_WT, i_A, i_B = self.init_cond

        time_to_fixation = 0
        while self._fixation_condition(i_WT, i_A, i_B):
            tau, i_WT, i_A, i_B = self.one_step_gillespie(i_WT, i_A, i_B)
            time_to_fixation += tau

        if i_WT == self.N:
            fixed_species = 'WT'
        elif i_A == self.N:
            fixed_species = 'A'
        else:
            fixed_species = 'B'

        return ({
            'time': int(time_to_fixation),
            'state': fixed_species})

    def _fixation_condition(self, i_WT, i_A, i_B):
        if i_WT == self.N:
            cond = False
        elif i_A == self.N:
            cond = False
        elif i_B == self.N:
            cond = False
        else:
            cond = True

        return cond

    def simulate_fixation(self, parameters):
        r"""
        Computes the number of each type of cell in a given tumor until
        fixation.

        Parameters
        ----------
        parameters
            (list) List of quantities that characterise the STEM cells cycle in
            this order: the initial counts for each type of cell (i_WT, i_A,
            i_B), the growth rate for the WT, the boosts in selection given to
            the mutated A and B variant respectively and the mutation rates
            with which a WT cell transforms into an A and B variant,
            respectively.

        """
        # Check correct format of output
        self._check_parameters_format(parameters)
        self._set_parameters(parameters)

        sol = self.gillespie_algorithm_fixation()

        computation_time = sol['time']
        fixed_species = sol['state']

        return computation_time, fixed_species


class StemGillespieTIMEVAR(StemGillespie):
    r"""StemGillespieTIMEVAR Class:
    Base class for the forward simulation of the evolution of a population
    of STEM cells.

    Three types of cells are considered - those which present mutations that
    give selectional advantage irrespective of environmental conditions (A),
    those which present mutations that give selectional advantage dependent
    on environmental conditions (B) and the wild type cells (WT).

    Cells decay at the same rate independent of their type and devide with
    rates which illsutate their selectional advantage. A wild type cell (WT)
    can mutate to a cell of type A, respectively a cell of type B with constant
    given rates of mutation.

    The system of equations that describe the isolated possible events that can
    occur

    .. math::
        :nowrap:

        \begin{eqnarray}
            WT &\xrightarrow{m} \emptyset \\
            A  &\xrightarrow{m} \emptyset \\
            B  &\xrightarrow{m} \emptyset \\
            \emptyset  &\xrightarrow{\alpha_{WT}} WT \\
            \emptyset  &\xrightarrow{\alpha_{A}} A \\
            \emptyset  &\xrightarrow{\alpha_{B}} B \\
            WT  &\xrightarrow{\mu_{A}} A \\
            WT  &\xrightarrow{\mu_{B}} B
        \end{eqnarray}

    where m is the rate of decay, :math:`\alpha_{WT}`, :math:`\alpha_{A}`,
    and :math:`\alpha_{B}` are the growth rates for the WT, A and B cell
    type respectively and :math:`\mu_{A}` and :math:`\mu_{B}` are the rate
    of mutation of a WT cell into A cell and respectively, a B cell type.
    For this class we consider the temporal selectional advatange of the
    B cells to vary with time.

    The total cell population is considered constant so the division of a cell
    is always simultaneous to the death of a cell.

    Therefore, the actual system of equations that describes the model is

    .. math::
        :nowrap:

        \begin{eqnarray}
            WT + WT &\xrightarrow{P_{WT \rightarrow A}} WT + A \\
            WT + WT &\xrightarrow{P_{WT \rightarrow B}} WT + B \\
            A + WT &\xrightarrow{P_{A \rightarrow B}} B + WT \\
            A + WT &\xrightarrow{P_{A \rightarrow WT}} WT + WT \\
            B + WT &\xrightarrow{P_{B \rightarrow A}} A + WT \\
            A + WT &\xrightarrow{P_{WT \rightarrow B}} A + B \\
            B + WT &\xrightarrow{P_{B \rightarrow WT}} WT + WT \\
            B + WT &\xrightarrow{P_{WT \rightarrow A}} A + B \\
            A + B &\xrightarrow{P_{A \rightarrow WT}} B + WT \\
            A + B &\xrightarrow{P_{B \rightarrow WT}} A + WT
        \end{eqnarray}

    """
    def __init__(self):
        super().__init__()

    def _prob_WT_to_B(self, t, i_WT, i_A, i_B):
        """
        Computes the probability of losing a WT cell and gaining a B cell
        when there is a change in the counts of the
        different species of cell in the tumor.

        This event can occur either through the mutation of a WT to a B,
        or the simultaneous division of a B cell which kills a WT cell.

        Parameters
        ----------
        t
            (int) time point at which we compute the transition probability.
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
            self.alpha_A * i_A) + (self.alpha_WT + (
                self.alpha_B-self.alpha_WT) * self._environment(t)) * i_B
        prob_B_divide = ((self.alpha_WT + (
                self.alpha_B-self.alpha_WT) * self._environment(t)
                ) * i_B) / tot_growth_rate
        divis = (1-mu) * prob_WT_die * prob_B_divide

        # Compute probability of change through mutation
        prob_WT_divide = (self.alpha_WT * i_WT) / tot_growth_rate
        mutat = self.mu_B * prob_WT_die * prob_WT_divide

        return (divis + mutat)

    def _prob_WT_to_A(self, t, i_WT, i_A, i_B):
        """
        Computes the probability of losing a WT cell and gaining a A cell
        when there is a change in the counts of the
        different species of cell in the tumor.

        This event can occur either through the mutation of a WT to a A,
        or the simultaneous division of a A cell which kills a WT cell.

        Parameters
        ----------
        t
            (int) time point at which we compute the transition probability.
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
            self.alpha_A * i_A) + (self.alpha_WT + (
                self.alpha_B-self.alpha_WT) * self._environment(t)) * i_B
        prob_A_divide = (self.alpha_A * i_A) / tot_growth_rate
        divis = (1-mu) * prob_WT_die * prob_A_divide

        # Compute probability of change through mutation
        prob_WT_divide = (self.alpha_WT * i_WT) / tot_growth_rate
        mutat = self.mu_A * prob_WT_die * prob_WT_divide

        return (divis + mutat)

    def _prob_B_to_WT(self, t, i_WT, i_A, i_B):
        """
        Computes the probability of losing a B cell and gaining a WT cell
        when there is a change in the counts of the
        different species of cell in the tumor.

        This event can only occur either through the simultaneous division of
        a WT cell which kills a B cell.

        Parameters
        ----------
        t
            (int) time point at which we compute the transition probability.
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
            self.alpha_A * i_A) + (self.alpha_WT + (
                self.alpha_B-self.alpha_WT) * self._environment(t)) * i_B
        prob_WT_divide = (self.alpha_WT * i_WT) / tot_growth_rate
        divis = (1-mu) * prob_B_die * prob_WT_divide

        return divis

    def _prob_A_to_WT(self, t, i_WT, i_A, i_B):
        """
        Computes the probability of losing a A cell and gaining a WT cell
        when there is a change in the counts of the
        different species of cell in the tumor.

        This event can only occur either through the simultaneous division of
        a WT cell which kills a A cell.

        Parameters
        ----------
        t
            (int) time point at which we compute the transition probability.
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
            self.alpha_A * i_A) + (self.alpha_WT + (
                self.alpha_B-self.alpha_WT) * self._environment(t)) * i_B
        prob_WT_divide = (self.alpha_WT * i_WT) / tot_growth_rate
        divis = (1-mu) * prob_A_die * prob_WT_divide

        return divis

    def _prob_A_to_B(self, t, i_WT, i_A, i_B):
        """
        Computes the probability of losing a A cell and gaining a B cell
        when there is a change in the counts of the
        different species of cell in the tumor.

        This event can only occur either through the simultaneous division of
        a B cell which kills a A cell.

        Parameters
        ----------
        t
            (int) time point at which we compute the transition probability.
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
            self.alpha_A * i_A) + (self.alpha_WT + (
                self.alpha_B-self.alpha_WT) * self._environment(t)) * i_B
        prob_B_divide = ((self.alpha_WT + (
                self.alpha_B-self.alpha_WT) * self._environment(t)
                ) * i_B) / tot_growth_rate
        divis = (1-mu) * prob_A_die * prob_B_divide

        return divis

    def _prob_B_to_A(self, t, i_WT, i_A, i_B):
        """
        Computes the probability of losing a B cell and gaining a A cell
        when there is a change in the counts of the
        different species of cell in the tumor.

        This event can only occur either through the simultaneous division of
        a A cell which kills a B cell.

        Parameters
        ----------
        t
            (int) time point at which we compute the transition probability.
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
            self.alpha_A * i_A) + (self.alpha_WT + (
                self.alpha_B-self.alpha_WT) * self._environment(t)) * i_B
        prob_A_divide = (self.alpha_A * i_A) / tot_growth_rate
        divis = (1-mu) * prob_B_die * prob_A_divide

        return divis

    def _environment(self, t):
        """
        Returns the environmental levels at a given time for the
        environment-dependent growth rate of the B cells using the matrix
        of switches embedded in the model.

        Parameters
        ----------
        t
            (int) time point at which we compute the environmental level.

        """
        # Find row in switches whose time immediately preceeds time t
        envir = self.switches[-1, 1]

        for _ in range(self.switches.shape[0]-1):
            if (self.switches[_, 0] <= t) and (t < self.switches[_+1, 0]):
                envir = self.switches[_, 1]

        return envir

    def one_step_gillespie(self, t, i_WT, i_A, i_B):
        """
        Computes one step in the Gillespie algorithm to determine the
        counts of the different species of cells living in the tumor at
        present. Returns time to next reaction and the tuple state of the
        system.

        Parameters
        ----------
        t
            (int) time point at which we compute the transition probability.
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
        # Generate random number for reaction and time to next reaction
        u, u1 = uniform.rvs(size=2)

        # Time to next reaction
        tau = np.log(1/u1)

        # Compute propensities
        propens_1 = self._prob_WT_to_B(t, i_WT, i_A, i_B)
        propens_2 = self._prob_WT_to_A(t, i_WT, i_A, i_B)
        propens_3 = self._prob_B_to_WT(t, i_WT, i_A, i_B)
        propens_4 = self._prob_A_to_WT(t, i_WT, i_A, i_B)
        propens_5 = self._prob_A_to_B(t, i_WT, i_A, i_B)
        propens_6 = self._prob_B_to_A(t, i_WT, i_A, i_B)

        propens = np.array([
            propens_1, propens_2, propens_3, propens_4,
            propens_5, propens_6])
        sum_propens = np.empty(propens.shape)
        for e in range(propens.shape[0]):
            sum_propens[e] = np.sum(propens[:(e+1)]) / np.sum(propens)

        if u < sum_propens[0]:
            i_WT += -1
            i_B += 1
        elif (u >= sum_propens[0]) and (u < sum_propens[1]):
            i_WT += -1
            i_A += 1
        elif (u >= sum_propens[1]) and (u < sum_propens[2]):
            i_WT += 1
            i_B += -1
        elif (u >= sum_propens[2]) and (u < sum_propens[3]):
            i_WT += 1
            i_A += -1
        elif (u >= sum_propens[3]) and (u < sum_propens[4]):
            i_B += 1
            i_A += -1
        elif (u >= sum_propens[4]) and (u < sum_propens[5]):
            i_A += 1
            i_B += -1

        return (tau, i_WT, i_A, i_B)

    def gillespie_algorithm_fixed_times(self, start_time, end_time):
        """
        Runs the Gillespie algorithm for the STEM cell population
        for the given times.

        Parameters
        ----------
        start_time
            (int) Time from which we start the simulation of the tumor.
        end_time
            (int) Time at which we end the simulation of the tumor.

        """
        # Create timeline vector
        times = np.arange(start_time, end_time+0.5, 1, dtype=np.int)
        interval = end_time - start_time + 1

        # Split compartments into their types
        i_WT, i_A, i_B = self.init_cond

        large_solution = []
        time_solution = []
        solution = np.empty((interval, 3), dtype=np.int)
        current_time = start_time
        while current_time <= end_time:
            time_solution.append(current_time)
            large_solution.append([i_WT, i_A, i_B])
            tau, i_WT, i_A, i_B = self.one_step_gillespie(
                current_time, i_WT, i_A, i_B)
            current_time += tau

        eval_indices = np.where(
            np.array([(t in time_solution) for t in times]))[0].tolist()

        ind_in_times = []
        j = 0
        for i, t in enumerate(eval_indices):
            if t < eval_indices[-1]:
                ind_in_times.extend([j]*(eval_indices[i+1]-eval_indices[i]))
            else:
                ind_in_times.extend([j]*(times[-1]-eval_indices[i]))
            j += 1

        for t in range(interval):
            solution[t, :] = large_solution[ind_in_times[t]]

        return({'state': solution})

    def simulate_fixed_times(
            self, parameters, switch_times, start_time, end_time):
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
        switch_times
            (list of lists) Array of the times at which the environmental
            conditions accounted for the B cell line. The first column
            indicates the time of change and the second indicate the level
            of the environment -- 0 for LOW; 1 for HIGH.
        start_time
            (int) Time from which we start the simulation of the tumor.
        end_time
            (int) Time at which we end the simulation of the tumor.

        """
        # Check correct format of output
        self._check_times(start_time, end_time)

        self._check_parameters_format(parameters)
        self._set_parameters(parameters)

        self._check_switch_times(switch_times)
        self.switches = np.asarray(switch_times)

        sol = self.gillespie_algorithm_fixed_times(start_time, end_time)

        output = sol['state']

        return output

    def _check_switch_times(self, switch_times):
        """
        Checks the format of the `switch_times` input in the simulation
        methods.

        """
        if np.asarray(switch_times).ndim != 2:
            raise ValueError(
                'Times of switches in environmental levels storage format must \
                    be 2-dimensional'
                )
        if np.asarray(switch_times).shape[1] != 2:
            raise ValueError(
                'Times of switches in environmental levels storage format must \
                    have 2 columns'
                )
        for _ in range(np.asarray(switch_times).shape[0]):
            if not isinstance(switch_times[_][0], int):
                raise TypeError('Times of switches in environmental levels must \
                    be integer.')
            if switch_times[_][0] < 0:
                raise ValueError('Times of switches in environmental levels must \
                    be => 0.')
            if not isinstance(
                    switch_times[_][1], (int, float)):
                raise TypeError('Environmental levels must \
                    be integer or float.')
            if switch_times[_][1] < 0:
                raise ValueError('Environmental levels must \
                    be => 0.')
        if switch_times[0][0] != 0:
            raise ValueError('First time of switch in environmental levels must \
                be => 0.')

    def gillespie_algorithm_criterion(self, criterion):
        """
        Runs the Gillespie algorithm for the STEM cell population
        until a criterion is met.

        Parameters
        ----------
        criterion
            (list of 2 lists) List of percentage thresholds of cell types in
            the population for disease to be triggered and another containing
            the type of threshold imposed.

        """
        # Split compartments into their types
        i_WT, i_A, i_B = self.init_cond

        time_to_criterion = 0
        while all(self._evaluate_criterion(criterion, i_WT, i_A, i_B)):
            tau, i_WT, i_A, i_B = self.one_step_gillespie(
                time_to_criterion, i_WT, i_A, i_B)
            time_to_criterion += tau

        return ({
            'time': int(time_to_criterion),
            'state': np.array([i_WT, i_A, i_B], dtype=np.int)})

    def simulate_criterion(self, parameters, switch_times, criterion):
        r"""
        Computes the number of each type of cell in a given tumor until a
        criterion is met.

        Parameters
        ----------
        parameters
            (list) List of quantities that characterise the STEM cells cycle in
            this order: the initial counts for each type of cell (i_WT, i_A,
            i_B), the growth rate for the WT, the boosts in selection given to
            the mutated A and B variant respectively and the mutation rates
            with which a WT cell transforms into an A and B variant,
            respectively.
        switch_times
            (list of lists) Array of the times at which the environmental
            conditions accounted for the B cell line. The first column
            indicates the time of change and the second indicate the level
            of the environment -- 0 for LOW; 1 for HIGH.
        criterion
            (list of 2 lists) List of percentage thresholds of cell types in
            the population for disease to be triggered and another containing
            the type of threshold imposed.

        """
        # Check correct format of output
        self._check_criterion(criterion)

        self._check_parameters_format(parameters)
        self._set_parameters(parameters)

        self._check_switch_times(switch_times)
        self.switches = np.asarray(switch_times)

        sol = self.gillespie_algorithm_criterion(criterion)

        computation_time = sol['time']
        final_state = sol['state']

        return computation_time, final_state

    def gillespie_algorithm_fixation(self):
        """
        Runs the Gillespie algorithm for the STEM cell population
        until fixation.

        """
        # Split compartments into their types
        i_WT, i_A, i_B = self.init_cond

        time_to_fixation = 0
        while self._fixation_condition(i_WT, i_A, i_B):
            tau, i_WT, i_A, i_B = self.one_step_gillespie(
                time_to_fixation, i_WT, i_A, i_B)
            time_to_fixation += tau

        if i_WT == self.N:
            fixed_species = 'WT'
        elif i_A == self.N:
            fixed_species = 'A'
        else:
            fixed_species = 'B'

        return ({
            'time': int(time_to_fixation),
            'state': fixed_species})

    def simulate_fixation(self, parameters, switch_times):
        r"""
        Computes the number of each type of cell in a given tumor until
        fixation.

        Parameters
        ----------
        parameters
            (list) List of quantities that characterise the STEM cells cycle in
            this order: the initial counts for each type of cell (i_WT, i_A,
            i_B), the growth rate for the WT, the boosts in selection given to
            the mutated A and B variant respectively and the mutation rates
            with which a WT cell transforms into an A and B variant,
            respectively.
        switch_times
            (list of lists) Array of the times at which the environmental
            conditions accounted for the B cell line. The first column
            indicates the time of change and the second indicate the level
            of the environment -- 0 for LOW; 1 for HIGH.

        """
        # Check correct format of output
        self._check_parameters_format(parameters)
        self._set_parameters(parameters)

        self._check_switch_times(switch_times)
        self.switches = np.asarray(switch_times)

        sol = self.gillespie_algorithm_fixation()

        computation_time = sol['time']
        fixed_species = sol['state']

        return computation_time, fixed_species
