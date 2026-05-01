#include <math.h>
#include "PRNG.h"
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <vector>


//Struct to easily return results on greeks and price.
struct OptionResults {
    double price;
    double delta;
    double vega;
    double time_taken;
};

//Function definition for monte carlo simulation
double monteCarloAsianSerial(int samples, double St0, double strike, double sigma, double r, double T);

double monteCarloAsianParallel(int samples, double St0, double strike, double sigma, double r, double T);

double getAsianPrice(int samples, double St0, double strike, double sigma, double r, double T);

OptionResults monteCarloAsianGreeksParallel(int samples, double St0, double strike, double sigma, double r, double T);

int main(){
    //Example inputs
    int samples = 5000000;       //Number of simulation paths
    double St0 = 100.0;          //Initial stock price (S_0)
    double strike = 103.0;       //Strike price (K)
    double sigma = 0.10;         //Volatility
    double r = 0.01;             //Risk-free interest rate 
    double T = 255.0;


    std::cout << "Running Monte Carlo simulation" << std::endl;
    std::cout << "Samples: " << samples << std::endl;

    //Call the function and store the result
    double startAsianSerial = omp_get_wtime();
    double estimatedAsianPriceSerial = monteCarloAsianSerial(samples, St0, strike, sigma, r, T);
    double endAsianSerial = omp_get_wtime();
    //double valueDiffAsianSerial = abs(expectedValue - estimatedPriceSerial);
    double serialAsianTime = endAsianSerial - startAsianSerial;

    double startAsianParallel = omp_get_wtime();
    double estimatedAsianPriceParallel = monteCarloAsianSerial(samples, St0, strike, sigma, r, T);
    double endAsianParallel = omp_get_wtime();
    //double valueDiffAsianSerial = abs(expectedValue - estimatedPriceSerial);
    double ParallelAsianTime = endAsianParallel - startAsianParallel;

    OptionResults finalResults = monteCarloAsianGreeksParallel(samples, St0, strike, sigma, r, T);

    

    

    std::cout << "-----------------------------------" << std::endl;
    std::cout << "Price: €" << finalResults.price << std::endl;
    std::cout << "Delta:  " << finalResults.delta << std::endl;
    std::cout << "Vega:   " << finalResults.vega << std::endl;
    std::cout << "Took:   " << finalResults.time_taken << " seconds." << std::endl;
    std::cout << "Estimated Option Price (Serial): €" << estimatedAsianPriceSerial << " Took: " << serialAsianTime << " seconds. Difference from expected value = " << std::endl;
    std::cout << "Estimated Option Price (Parallel): €" << estimatedAsianPriceParallel << " Took: " << ParallelAsianTime << " seconds. Difference from expected value = " << std::endl;
   
    return 0;
}


//Serial Monte Carlo implementation.
double monteCarloAsianSerial(int samples, double St0, double strike, double sigma, double r, double T){
    double dt = 1.0 / 255.0;
    double mu_t = (r - 0.5 * sigma * sigma) * dt;
    double step_vol = sigma * sqrt(dt);
    double total_payoff = 0.0;

    PRNG myGenerator(1);

    //Loop for each path
    for(int i = 0; i < samples; ++i){
        double current_S = St0;
        double sum_S = St0; //Include starting price in the running sum
        
        //Loop through individual days on each path
        for(int t = 1; t <= T; ++t){
            double randVariate = myGenerator.getStandardNormal();

            //Calculate current days value
            current_S *= exp(mu_t + step_vol * randVariate);

            
            sum_S += current_S;//Sum to total
        }

        //Caculate average for 255+1 as includes final day so 0-> 255 days
        double A_T = sum_S / (T + 1.0);

        //Caculate path payoff.
        double path_payoff = std::max(A_T - strike, 0.0);
        total_payoff += path_payoff;
    }
    

    total_payoff = total_payoff/samples;

    //Could assume year is just 1 if always using 255 days but here to allow change
    double T_years = T / 255.0; 
    double payoff_value = exp(-r * T_years) * total_payoff;

    return payoff_value;
}

double monteCarloAsianParallel(int samples, double St0, double strike, double sigma, double r, double T) {
    double dt = 1.0 / 255.0;
    double mu_t = (r - 0.5 * sigma * sigma) * dt;
    double step_vol = sigma * sqrt(dt);
    double total_payoff = 0.0;


    //Use reduction to add up total_payoff across all threads
    #pragma omp parallel reduction(+:total_payoff)
    {

        int thread_id = omp_get_thread_num();
        //Initialize one generator per thread. 
        PRNG myGenerator(1 + thread_id); 

        //Divide the 1,000,000 loop iterations among the threads
        #pragma omp for 
        for(int i = 0; i < samples; ++i) {
            double current_S = St0;
            double sum_S = St0; 
            
            for(int t = 1; t <= T; ++t) {
                double randVariate = myGenerator.getStandardNormal();
                current_S *= exp(mu_t + step_vol * randVariate);
                sum_S += current_S; 
            }

            double A_T = sum_S / (T + 1.0);
            double path_payoff = std::max(A_T - strike, 0.0);
            
            total_payoff += path_payoff;
        }
    }
    
    total_payoff = total_payoff / samples;

    //Discount back to present value
    double T_years = T / 255.0; 
    double payoff_value = exp(-r * T_years) * total_payoff;

    return payoff_value;
}

double getAsianPrice(int samples, double St0, double strike, double sigma, double r, double T) {
    double dt = 1.0 / 255.0;
    double mu_t = (r - 0.5 * sigma * sigma) * dt;
    double step_vol = sigma * sqrt(dt);
    double total_payoff = 0.0;

    #pragma omp parallel for reduction(+:total_payoff)
    for(int i = 0; i < samples; ++i) {
        
        //Because the seed is tied to 'i' identical inputs yield identical paths.
        PRNG myGenerator(1 + i); 

        double current_S = St0;
        double sum_S = St0;
        
        for(int t = 1; t <= T; ++t) {
            current_S *= exp(mu_t + step_vol * myGenerator.getStandardNormal());
            sum_S += current_S; 
        }

        double A_T = sum_S / (T + 1.0);
        total_payoff += std::max(A_T - strike, 0.0);
    }
    
    return exp(-r * (T / 255.0)) * (total_payoff / samples);
}

OptionResults monteCarloAsianGreeksParallel(int samples, double St0, double strike, double sigma, double r, double T) {
    OptionResults results;
    double start_time = omp_get_wtime();

    //Get Base Price
    results.price = getAsianPrice(samples, St0, strike, sigma, r, T);

    //Calculate Delta (Bump Price by 1%)
    double dS = St0 * 0.01; 
    double price_s_up   = getAsianPrice(samples, St0 + dS, strike, sigma, r, T);
    double price_s_down = getAsianPrice(samples, St0 - dS, strike, sigma, r, T);
    results.delta = (price_s_up - price_s_down) / (2.0 * dS);

    //Calculate Vega (Bump Volatility by 0.1%)
    double dSig = 0.001; 
    double price_v_up   = getAsianPrice(samples, St0, strike, sigma + dSig, r, T);
    double price_v_down = getAsianPrice(samples, St0, strike, sigma - dSig, r, T);
    results.vega = (price_v_up - price_v_down) / (2.0 * dSig);

    double end_time = omp_get_wtime();
    results.time_taken = end_time - start_time;

    return results;
}
