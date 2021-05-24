import numpy as np

def main():
    


#include <iostream>
#include <thread>

#include "simulator.h"
#include "params.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/filesystem.hpp>

using namespace std;

void run_task(int threadID, sim_params simParams, std::vector<mc_params> mcParams) {

    int n_runs = mcParams.size();
    for(int i = 0; i < n_runs; ++i) {

        auto start = std::chrono::high_resolution_clock::now();

        long seed = mcParams[i].seed;
        long simulationID = mcParams[i].simulationID;

        simulator simulator("run_" + to_string(simulationID), &simParams, mcParams[i]);

        simulator.run();

        cout << "SIMULATION >> ID: " << mcParams[i].simulationID;
        cout << " (SEED:" << seed << "; THREAD:" << threadID << ")\n";
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        cout << "Simulation completed in " << duration.count() / 1000.0 << "s.\n" << endl;
    }
}


/**
 * Market Simulator.
 */
int main(int argc, char** argv) {

    // TODO: throw specific exception about too many arguments. single argument expected.
    if (argc > 2)
        throw simuException();

    const string &WORKFLOW = "../Workflow/ZI";
    const string &RESULT_PATH = "../Workflow/ZI/Results";

    std::cout << "--- MARKET SIMULATION ---" << std::endl;

    boost::property_tree::ptree p;
    boost::property_tree::read_json(WORKFLOW + "/sim_params.json", p);

    auto symbol = p.get<string>("symbol");

    // Hard-coded simulation parameters.
    unsigned long int step_size = 20000L;                   // step size in microseconds
    unsigned long int n_steps = 30600000000L / step_size;   // number of steps for a full trading day
    int n_threads = 6;
    int n_runs = 1;
    int verbose = 3;

    // Hard-coded strategy parameters.
    double alpha = 0.021512;
    double mu = 0.00134;
    double delta = 0.000531;
    double lambda = 0.04;

    sim_params simParams = {
            .symbol = symbol,
            .closing_bid_prc = p.get<double>("closing_bid_prc"),
            .closing_ask_prc = p.get<double>("closing_ask_prc"),
            .tick_size = p.get<float>("tick_size"),
            .n_steps = n_steps,
            .step_size = step_size,
            .n_threads = n_threads,
            .n_runs = p.get<int>("n_runs", 1),
            .verbose = verbose,
            .l2_depth = (unsigned int) p.get<double>("l2_depth", 10),
            .tick_format = p.get<string>("tick_format", ""),
            .date_format = p.get<string>("date_format", "%d-%b-%Y %H:%M:%S"),
            .seed = p.get<long>("seed", 1234L),
            .n_traders = p.get<unsigned int>("n_traders", 100)
    };

    std::default_random_engine generator(simParams.seed);

    auto *seed_dist = new std::uniform_int_distribution<int>(0, INT_MAX);

    vector<mc_params> mcParamsVec;
    for(int run = 0; run < simParams.n_runs; ++run) {
        mc_params mcParams = {
                .simulationID = run,
                .seed = (long) (*seed_dist)(generator),
                .results_path = WORKFLOW + "/Results/" + to_string(run),
                .ziParams = {
                    .alpha = alpha / simParams.n_traders,
                    .mu = mu / simParams.n_traders,
                    .delta = delta,
                    .lambda = lambda
                }
        };

        mcParamsVec.push_back(mcParams);
    }

    unsigned int _n_threads = min(min(simParams.n_threads, simParams.n_runs), (int)std::thread::hardware_concurrency());

    auto start = std::chrono::high_resolution_clock::now();
    cout << "MONTE-CARLO SIMULATION (N_THREADS:" << n_threads << ", MC_RUNS:" << simParams.n_runs << "):\n\n";

    std::vector<mc_params> _mcParams[_n_threads];
    for (int i = 0; i < simParams.n_runs; ++i)
        _mcParams[i % n_threads].push_back(mcParamsVec[i]);

    std::thread threadPool[_n_threads];
    for (int i = 0; i < _n_threads; ++i)
        threadPool[i] = std::thread(run_task, i, simParams, _mcParams[i]);

    for (int i = 0; i < _n_threads; ++i)
        threadPool[i].join();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    cout << "Monte-carlo simulation completed in " << duration.count() / 1000.0 << "s." << endl;
}
