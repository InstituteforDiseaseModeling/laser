import numpy as np
import unittest
from copy import deepcopy
import pdb
import sys
sys.path.append( "." )
import settings
import report
from sir_numpy import load, collect_report

def collect_and_report(ctx, csvwriter, timestep):
    currently_infectious, currently_sus, cur_reco = collect_report( ctx )
    counts = {
            "S": deepcopy( currently_sus ),
            "I": deepcopy( currently_infectious ),
            "R": deepcopy( cur_reco ) 
        }
    def normalize( sus, inf, rec ):
        totals = {}
        for idx in currently_sus.keys():
            totals[ idx ] = sus[ idx ] + inf[ idx ] + rec[ idx ]
            sus[ idx ] /= totals[ idx ] 
            inf[ idx ] /= totals[ idx ] 
            rec[ idx ]/= totals[ idx ] 
        return totals
    totals = normalize( currently_sus, currently_infectious, cur_reco )
    fractions = {
            "S": currently_sus,
            "I": currently_infectious,
            "R": cur_reco 
        }
    report.write_timestep_report( csvwriter, timestep, counts["I"], counts["S"], counts["R"] )
    return counts, fractions, totals


class TestCalculateNewInfections(unittest.TestCase):

    def test_calculate_new_infections_np(self):
        from model_numpy import calculate_infections as calc
        data = load( settings.pop_file )
        csv_writer = report.init()
        counts, fractions, totals = collect_and_report(data, csv_writer,0)

        result = calc.calculate_new_infections(data, fractions["I"], fractions["S"], totals, settings.base_infectivity)
        # Add your assertions here to check the correctness of the result
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 1)
        self.assertTrue(all(isinstance(val, np.uint32) for val in result))
        self.assertEqual(result[0], 0)
        # Add more specific assertions based on the expected behavior of the function

        data["incubation_timer"][:] = 0 # end incubation
        counts, fractions, totals = collect_and_report(data, csv_writer,0)

        result = calc.calculate_new_infections(data, fractions["I"], fractions["S"], totals, settings.base_infectivity)
        # Add your assertions here to check the correctness of the result
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 1)
        self.assertTrue(all(isinstance(val, np.uint32) for val in result))
        self.assertEqual(result[0], 99)
        # Add more specific assertions based on the expected behavior of the function

    def test_calculate_new_infections_c(self):
        from model_numpy_c import calculate_infections as calc
        from sir_numpy_c import load, collect_report
        data = load( settings.pop_file )
        csv_writer = report.init()
        counts, fractions, totals = collect_and_report(data, csv_writer,0)

        result = calc.calculate_new_infections(data, fractions["I"], fractions["S"], totals, settings.base_infectivity)
        # Add your assertions here to check the correctness of the result
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 1)
        self.assertTrue(all(isinstance(val, np.uint32) for val in result))
        self.assertEqual(result[0], 0)
        # Add more specific assertions based on the expected behavior of the function

        data["incubation_timer"][:] = 0
        counts, fractions, totals = collect_and_report(data, csv_writer,0)

        result = calc.calculate_new_infections(data, fractions["I"], fractions["S"], totals, settings.base_infectivity)
        # Add your assertions here to check the correctness of the result
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 1)
        self.assertTrue(all(isinstance(val, np.uint32) for val in result))
        self.assertTrue(result[0], 99)
        # Add more specific assertions based on the expected behavior of the function

if __name__ == '__main__':
    unittest.main()

