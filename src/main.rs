#![allow(non_snake_case, dead_code, unused_assignments, non_camel_case_types)]
use integer_partitions::Partitions;
use noisy_float::prelude::*;
use rand::prelude::*;
use rand_distr::Exp;
use rand_distr::Gamma;
use rand_distr::Uniform;
use smallvec::{SmallVec, smallvec};
use std::f64::INFINITY;

const EPSILON: f64 = 1e-8;
const DEBUG: bool = false;

fn main() {
    println!("Lambda; Mean Response Time;");

    //let dist = Dist::Hyperexp(1.0,job_size_mu,0.5);
    //let dist = Dist::Gamma(3.0, 0.3);
    //let dist = Dist::Uniform(0.01,1.0);
    let dist = Dist::Expon(1.0);
    let num_servers = 1;
    let num_jobs = 1_000_000;
    let seed = 3;

    //homogenous job service requirement:
    //let job_req_dist = Dist::Constant(0.45);
    let job_req_dist = Dist::Uniform(0.0, 1.0);

    let policy = Policy::IPB(8);
    println!(
        "Policy : {:?}, Duration: {:?}, Requirement: {:?}, Jobs per data point: {}, Seed: {}",
        policy, dist, job_req_dist, num_jobs, seed
    );
    for lam_base in 1..20 {
        let lambda = lam_base as f64 / 10.0;
        let check = simulate(
            policy,
            num_servers,
            num_jobs,
            dist,
            lambda,
            seed,
            job_req_dist,
        );

        println!("{}; {};", lambda, check);
    }
}

#[derive(Debug)]
struct Job {
    arrival_time: f64,
    original_size: f64,
    rem_size: f64,
    service_req: f64,
}

// Make a distribution enum

#[derive(Debug, Clone, Copy)]
enum Dist {
    // various distribution functions
    Expon(f64),
    Hyperexp(f64, f64, f64),
    Gamma(f64, f64),
    Uniform(f64, f64),
    Constant(f64),
}

impl Dist {
    fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        // take a sample from a given distribution
        match self {
            Dist::Hyperexp(low_mu, high_mu, prob_low) => {
                let mu = if *prob_low == 1.0 {
                    low_mu
                } else if rng.r#gen::<f64>() < *prob_low {
                    low_mu
                } else {
                    high_mu
                };
                Exp::new(*mu).unwrap().sample(rng)
            }
            Dist::Expon(lambda) => Exp::new(*lambda).unwrap().sample(rng),

            Dist::Gamma(k, scale) => Gamma::new(*k, *scale).unwrap().sample(rng),
            Dist::Uniform(low, high) => Uniform::try_from(*low..*high).unwrap().sample(rng),
            Dist::Constant(val) => *val,
        }
    }
    fn mean(&self) -> f64 {
        // return the mean of a given distribution
        use Dist::*;
        match self {
            Hyperexp(low_mu, high_mu, prob_low) => prob_low / low_mu + (1.0 - prob_low) / high_mu,
            Expon(lambda) => 1.0 / lambda,
            Gamma(k, scale) => k * scale,
            Uniform(low, high) => (low + high) / 2.0,
            Constant(val) => *val,
        }
    }

    fn meansquare(&self) -> f64 {
        // return the mean square of a given distribution
        use Dist::*;
        match self {
            Hyperexp(low_mu, high_mu, prob_low) => {
                (2.0 / (low_mu.powf(2.0)) * prob_low)
                    + (2.0 / (high_mu.powf(2.0)) * (1.0 - prob_low))
            }
            Expon(lambda) => 2.0 / lambda.powf(2.0),
            Gamma(k, scale) => ((k + 1.0) * k) / (1.0 / scale).powf(2.0),
            Uniform(low, high) => (1.0 / 3.0) * ((high.powf(3.0) - low.powf(3.0)) / (low - high)),
            Constant(val) => *val,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Policy {
    // Baseline policies
    FCFS,       // First-Come First-Served
    PLCFS,      // Preemptive Last-Come First-Served
    SRPT,       // Shortest Remaining Processing Time
    FCFSB,      // First-Come First-Served, preemptive backfilling
    SRPTB,      // Shortest Remaining Processing Time, preemptive backfilling
    PLCFSB,     // Preemptive Last-Come First-Served
    LSF,        // Preemptive Least Servers First
    LSFB,       // Preemptive Least Servers First, preemptive backfilling
    MSF,        // Preemptive Most Servers First
    MSFB,       // Preemptive Most Servers First, preemptive backfilling
    SRA,        // Smallest remaining area
    SRAB,       // Smallest remaining area, preemptive backfilling
    LRA,        // Largest remaining area
    LRAB,       // Largest remaining area, preemptive backfilling
    DB(usize),  // Double bucket, explicit K
    DBB(usize), // Double bucket, explicit K, preemptive backfilling
    DBE,        // Double bucket, K based on lambda
    DBEB,       // Double bucket, K based on lambda, preemptive backfilling
    BPT(usize), // Bucket powers of 2, explicit K
    // TODO: BPT w/ backfilling
    AdaptiveDoubleBucket, // Double bucket, K based on queue length
    // TODO: Adaptive Double Bucket w/ backfilling
    IPB(usize), // Integer partitions buckets, explicit K
                // TODO: IPB w/ backfilling
}

impl Policy {
    // return whichever criterion jobs get sorted by.

    fn index(&self, job: &Job) -> f64 {
        match self {
            Policy::FCFS | Policy::FCFSB => job.arrival_time,
            Policy::PLCFS | Policy::PLCFSB => -job.arrival_time,
            Policy::SRPT | Policy::SRPTB => job.rem_size,
            Policy::LSF | Policy::LSFB => job.service_req,
            Policy::MSF | Policy::MSFB => -job.service_req,
            Policy::SRA | Policy::SRAB => job.rem_size * job.service_req,
            Policy::LRA | Policy::LRAB => -job.rem_size * job.service_req,
            Policy::DB(_)
            | Policy::DBE
            | Policy::DBB(_)
            | Policy::DBEB
            | Policy::BPT(_)
            | Policy::AdaptiveDoubleBucket
            | Policy::IPB(_) => job.arrival_time,
        }
    }
}

fn fcfstest(arr_lambda: f64, size_dist: &Dist) {
    // if FCFS is being used, can predict expected behavior of queueing system
    // used to ensure that simulation worked previously.

    let avg_size = size_dist.mean();

    // rho -- must be less than 1
    let rho = arr_lambda * avg_size;

    let esquare = size_dist.meansquare();

    // we have everything needed to find E[T] and E[N]
    let ET = (arr_lambda * esquare) / (2.0 * (1.0 - rho)) + avg_size;
    let EN = ET * arr_lambda;
    if DEBUG {
        println!("Mean Response time is: {}, Mean Queue Length is {}", ET, EN);
    }
}

fn qscan(vec: &Vec<Job>, num_servers: usize) -> usize {
    // only works for sorting-based policies.
    // iterate through a queue until the service requirement is maxed out.
    let mut index = 0;
    let total_resource = num_servers as f64;
    if DEBUG {
        println!("Total resource = {}", total_resource);
        println!("Queue length={}", vec.len());
    }
    // count how much "size" we have remaining in this timestep
    let mut taken_service: f64 = EPSILON;

    // very naive while loop
    while taken_service < total_resource {
        if index >= vec.len() {
            if DEBUG {
                println!("Max Length reached");
            }
            return vec.len();
        };
        taken_service = taken_service + vec[index].service_req;
        index = index + 1;
    }
    index - 1
}

fn take_to_vec(num_take: usize) -> Vec<usize> {
    // convert a usize to a collection of indices of jobs that can be worked on if qscan was used.
    let v: Vec<usize> = (0..num_take).collect();
    v
}

fn backfill(vec: &Vec<Job>, num_servers: usize) -> Vec<usize> {
    // backfilling that assumes you haven't chosen anything to work on yet.
    // works for sorting-based policies with backfilling
    let total_resource = num_servers as f64;
    if DEBUG {
        println!("Backfilling up to {}", total_resource);
    }

    // initialize the taken_resource counter, loop with a skip
    let mut taken_resource = 0.0;
    let mut indices: Vec<usize> = vec![];

    for ii in 0..vec.len() {
        let trial_take = taken_resource + vec[ii].service_req;
        if trial_take > total_resource {
            continue;
        }
        if trial_take + EPSILON <= total_resource {
            taken_resource = trial_take;
            indices.push(ii);
        }
    }
    indices
}

fn backfill_hogged(vec: &Vec<Job>, hogged: f64, hog_indices: Vec<usize>) -> Vec<usize> {
    // backfilling function that takes in a vector of indices that have already been chosen
    // by your policy. returns a vector of all indices that can be worked on after backfilling.
    let total_resource = 1.0;
    if DEBUG {
        println!("Backfilling up to {}", total_resource);
    }

    let mut taken_resource = hogged;
    let mut indices = hog_indices.clone();

    for ii in 0..vec.len() {
        if hog_indices.contains(&ii) {
            continue;
        }
        let trial_take = taken_resource + vec[ii].service_req;
        if trial_take > total_resource {
            continue;
        }
        if trial_take + EPSILON <= total_resource {
            taken_resource = trial_take;
            indices.push(ii);
        }
    }
    indices
}

// make the bucket giving a separate function for the IP policy to do its thing
fn assign_buckets(vec: &Vec<Job>, k: usize, upper: f64, lower: f64) -> Vec<usize> {
    // return a vector the same length as the queue,
    // containing the bucket label of the job at its corresponding index.
    let increment = (upper - lower) / (k as f64);
    if DEBUG {
        println!("Increment is {}, k is {}", increment, k);
    }
    let all_indices: Vec<usize> = (0..vec.len()).collect();

    let bucket_numbers: Vec<usize> = all_indices
        .iter()
        .map(|index| (vec[*index].service_req / increment).floor() as usize)
        .collect();
    bucket_numbers
}

fn eval_buckets(vec: &Vec<Job>, k: usize, upper: f64, lower: f64, backfill: bool) -> Vec<usize> {
    //Double bucket policy, forms and scores bucket pairs.
    assert!(k % 2 == 1);

    let bucket_numbers: Vec<usize> = assign_buckets(vec, k, upper, lower);
    // evaluate bucket scores
    let mut bucket_counts: Vec<f64> = vec![0.0; k];
    for ii in 0..vec.len() {
        bucket_counts[bucket_numbers[ii]] += 1.0;
    }
    // square all bucket scores
    let bucket_scores: Vec<f64> = bucket_counts.iter().map(|score| score.powf(2.0)).collect();

    // compare bucket scores and return the highest one
    let mut target = 0; // 0 corresponds to bucket pair 0,k-1
    let mut temp_new = 0.0;
    let mut sitting_best = 0.0;
    for jj in 0..((bucket_scores.len() - 1) / 2) {
        temp_new = bucket_scores[jj] + bucket_scores[k - jj - 2];

        if temp_new > sitting_best + EPSILON {
            sitting_best = temp_new;
            target = jj; // assign target var
        }
    }

    // check the last bucket
    let mut last = false;
    if bucket_scores[k - 1] > sitting_best {
        target = k - 1;
        last = true;
    }

    if DEBUG {
        println!("Bucket scores: {:?}", bucket_scores);
        println!("Bucket numbers of jobs: {:?}", bucket_numbers);
        println!("Last bucket targeted?: {:?}", last);
    }
    let mut ret_indices: Vec<usize> = vec![];

    // fetch the indices of the jobs corresponding to the winning bucket
    for kk in 0..vec.len() {
        if bucket_numbers[kk] == target {
            ret_indices.push(kk);
            break;
        }
    }

    for kk in 0..vec.len() {
        if !last {
            if bucket_numbers[kk] == k - target - 2 {
                ret_indices.push(kk);
                break;
            }
        }
    }

    // backfilling
    if backfill {
        let taken_resource: f64 = ret_indices
            .iter()
            .map(|index| vec[*index].service_req)
            .sum();
        ret_indices = backfill_hogged(vec, taken_resource, ret_indices);
    }

    if DEBUG {
        println!("Working on jobs {:?}", ret_indices);
    }

    ret_indices
}

fn get_d(c: usize) -> usize {
    // not needed
    // calculate the lowest power of two greater than c.
    if c == 1 {
        return 1;
    }
    let mut d: usize = 2;
    while d < c {
        d = 2 * d;
    }
    d
}

fn c_to_bucket_pair(c: usize) -> SmallVec<[usize; 2]> {
    // convert c (a number corresponding to a certain pair of buckets of powers of two)
    // eg. can return just [4], or [3,1] and other combinations that fit regardless of repetitions.

    let d = c.next_power_of_two(); // highest power of 2
    assert!((d & (d - 1)) == 0);

    let mut bucket_set = smallvec![c];
    if d - c != 0 {
        bucket_set.push(d - c);
    }
    bucket_set
}

fn p2_buckets(vec: &Vec<Job>, k: usize, backfill: bool) -> Vec<usize> {
    // assigns buckets of powers of two to each job in a queue and returns indices corresponding to
    // the highest-scoring set.

    assert!((k & (k - 1)) == 0);

    // bucket 1 is the smallest bucket
    let bucket_numbers: Vec<usize> = assign_buckets(vec, k, 1.0, 0.0)
        .iter()
        .map(|b| b + 1)
        .collect();

    // evaluate bucket set scores
    let mut set_scores: Vec<usize> = vec![0; k];

    let mut bucket_counts = vec![0; k];
    for &num in &bucket_numbers {
        bucket_counts[num - 1] += 1
    }
    /*
    for ii in 0..k {
        let c = ii + 1;
        // set 1 is the first set
        let bucket_set = c_to_bucket_pair(c);

        let reps: usize = k / c.next_power_of_two();

        // counts of buckets in bucket_set starting from bucket 1
        let q: Vec<usize> = bucket_set
            .iter()
            .map(|&c| bucket_numbers.iter().filter(|&num| *num == c).count())
            .collect();
        if DEBUG {
            println!("There are {:?} jobs of bucket {} in queue", q, c);
        }

        // calculate score of the iith set
        let mut score: usize = 0;
        for jj in 0..q.len() {
            score = score + q[jj].min(reps);
        }

        set_scores[ii] = score;
    }
    */
    for ii in 0..k {
        let c = ii + 1;
        // set 1 is the first set
        let bucket_set = c_to_bucket_pair(c);

        let reps: usize = k / c.next_power_of_two();

        for jj in 0..bucket_set.len() {
            let num_in_bucket = bucket_counts[bucket_set[jj] - 1];
            let num_jobs_served = num_in_bucket.min(reps);
            set_scores[ii] += num_in_bucket * num_jobs_served;
        }
    }

    // now we know the bucket scores. find the highest scoring set, then return non-repeating
    // indices to corresponding jobs

    let (s_i, _big_score) = set_scores
        .iter()
        .enumerate()
        .max_by_key(|(_index, score)| *score)
        .expect("At least one set");
    /*
    let big_score = *set_scores.iter().max().unwrap();

    //get the index of the top_scoring set.
    let s_i = set_scores
        .iter()
        .position(|x| x == &big_score)
        .expect("Top score not found");
        */

    let target_c = s_i + 1;
    let target_reps = k / target_c.next_power_of_two();
    let target_buckets = c_to_bucket_pair(target_c);

    if DEBUG {
        println!(
            "searching for bucket set with c value {}, repeating {} times, bucket values {:?}",
            target_c, target_reps, target_buckets
        );
    }

    let mut found_indices: Vec<usize> = vec![];

    // go through and find indices of buckets that match
    /*
    for _rep in 0..target_reps {
        for kk in 0..bucket_numbers.len() {
            let current = &bucket_numbers[kk];
            if target_buckets.contains(&current) & !found_indices.contains(&kk) {
                found_indices.push(kk);

            }
        }
    }

    for _rep in 0..target_reps {
        for target_b in target_buckets {
            for kk in 0..bucket_numbers.len() {
                let current = bucket_numbers[kk];

            }
        }
    }
    */

    for target_b in target_buckets {
        let mut count = 0;
        for kk in 0..bucket_numbers.len() {
            let current = bucket_numbers[kk];
            if target_b == current {
                found_indices.push(kk);
                count += 1;
                if count == target_reps {
                    break;
                }
            }
        }
    }

    if backfill {
        let taken_resource: f64 = found_indices
            .iter()
            .map(|index| vec[*index].service_req)
            .sum();
        found_indices = backfill_hogged(vec, taken_resource, found_indices);
    }
    // now we haave indices of the buckets to work on
    // TODO: write a test for this

    found_indices
}

#[derive(Debug, Clone)]
struct score_ip {
    partition: Vec<usize>,
    score: usize,
}

#[derive(Debug, Clone)]
struct scored_vec_mult {
    vect: Vec<usize>,
    multiplicities: Vec<usize>,
    score: usize,
}

fn k_to_partitions_mults(k: usize) -> Vec<scored_vec_mult> {
    // takes a k value and returns a vector of integer partitions with multiplicities of each
    // number listed and no duplicates (hopefully)

    let mut ipar = Partitions::new(k);
    let mut partition_vector: Vec<scored_vec_mult> = vec![];

    while let Some(part) = ipar.next() {
        let current_partition: Vec<usize> = part.to_vec();
        let mut no_duplicates: Vec<usize> = vec![];
        let mut mults: Vec<usize> = vec![];

        for ii in 0..current_partition.len() {
            let num = current_partition[ii];
            if no_duplicates.contains(&num) {
                continue;
            } else {
                no_duplicates.push(num);
                let multiplicity = current_partition.iter().filter(|&n| n == &num).count();
                mults.push(multiplicity)
            }
        }
        let current_set = scored_vec_mult {
            vect: no_duplicates,
            multiplicities: mults,
            score: 0,
        };
        partition_vector.push(current_set);
    }
    partition_vector
}

fn vec_mult_to_work(job_vec: &Vec<Job>, k: usize, sets: Vec<scored_vec_mult>, backfill: bool,) -> Vec<usize> {
    //smallest bucket is 1
    let bucket_numbers: Vec<usize> = assign_buckets(&job_vec, k, 1.0, 0.0)
        .iter()
        .map(|b| b + 1)
        .collect();
    if DEBUG {
        println!("Bucket numbers: {:?}", bucket_numbers)
    }
    let mut score_vec: Vec<scored_vec_mult> = sets.clone();

    // bucket_counts is the quantity of jobs in each bucket

    let mut bucket_counts = vec![0; k];
    for &num in &bucket_numbers {
        bucket_counts[num - 1] += 1
    }

    for qq in 0..score_vec.len() {
        let set = score_vec[qq].clone();
        let current_partition: Vec<usize> = set.vect;
        let mul_vec = set.multiplicities;
        let mut current_score: usize = 0;
        for ii in 0..current_partition.len() {
            let multiplicity = mul_vec[ii];
            let count = bucket_counts[ii];
            current_score += multiplicity.min(count);
        }
        score_vec[qq].score = current_score;
    }

    let score_vec_for_eval = score_vec.clone();

    let top_scorer: scored_vec_mult = score_vec_for_eval
        .iter()
        .max_by_key(|p: &&scored_vec_mult| p.score)
        .unwrap()
        .clone();
    let target_buckets = top_scorer.vect;
    let target_reps = top_scorer.multiplicities;
    assert!(target_buckets.len() == target_reps.len());

    if DEBUG {
        println!("Chosen partition: {:?} with multiplicities {:?}", target_buckets, target_reps);
    }
    let mut found_indices: Vec<usize> = vec![];

    for jj in 0..target_buckets.len() {
        let bucket_num = target_buckets[jj];
        let multiplicity = target_reps[jj];
        let mut count = 0;
        for kk in 0..bucket_numbers.len() {
            let current = bucket_numbers[kk];
            if bucket_num == current {
                found_indices.push(kk);
                count += 1;
                if count == multiplicity {
                    break;
                }
            }
        }
    }
    found_indices
}

fn ipar_buckets(vec: &Vec<Job>, k: usize, backfill: bool) -> Vec<usize> {
    // assigns buckets of powers of two to each job in a queue and returns indices corresponding to
    // the highest-scoring set.

    // smallest bucket is 1
    let bucket_numbers: Vec<usize> = assign_buckets(vec, k, 1.0, 0.0)
        .iter()
        .map(|b| b + 1)
        .collect();

    if DEBUG {
        println!("Bucket numbers: {:?}", bucket_numbers)
    }

    // evaluate bucket set scores
    let mut ip_scores: Vec<score_ip> = vec![];

    // bucket_counts is the quantity of jobs in each bucket

    let mut bucket_counts = vec![0; k];
    for &num in &bucket_numbers {
        bucket_counts[num - 1] += 1
    }
    // get integer partitions are score them

    let mut ipar = Partitions::new(k);

    while let Some(part) = ipar.next() {
        // get the current partition
        let current_partition: Vec<usize> = part.to_vec();
        let mut current_score: usize = 0;
        // iterate over the partition and match the counts to bucket_counts, then score
        let mut seen = vec![];
        for bucket_num in &current_partition {
            // only add score for new numbers because of the multiplicity calculation
            // (sorry)
            if seen.contains(&bucket_num) {
                continue;
            } else {
                seen.push(bucket_num);
            }
            // get multiplicity of each bucket number in the integer partition
            let multiplicity = current_partition
                .iter()
                .filter(|&num| num == bucket_num)
                .count();

            let count = bucket_counts[bucket_num - 1]; // number of bucket_num buckets we have
            current_score += multiplicity.min(count);
        }
        let pair = score_ip {
            partition: current_partition,
            score: current_score,
        };
        ip_scores.push(pair);
    }

    let top_scorer: score_ip = ip_scores
        .iter()
        .max_by_key(|p: &&score_ip| p.score)
        .unwrap()
        .clone();

    let target_buckets = top_scorer.partition;
    if DEBUG {
        println!("Chosen partition: {:?}", target_buckets);
    }
    let mut found_indices: Vec<usize> = vec![];

    // thisll be less efficient than the powers of two one

    /*
    for ii in 0..bucket_numbers.len() {
        for jj in found_count..target_buckets.len() {
           if target_buckets[jj] == bucket_numbers[ii] {
               found_indices.push(ii);
               found_count += 1;
               break;
           }
           else {
               continue;
           }
        }
    }
    */

    let mut seen = vec![];

    for bucket_num in &target_buckets {
        if seen.contains(&bucket_num) {
            continue;
        } else {
            seen.push(bucket_num);
        }
        // get multiplicity of each bucket number in the integer partition
        let multiplicity = target_buckets
            .iter()
            .filter(|&num| num == bucket_num)
            .count();
        let mut count = 0;
        for kk in 0..bucket_numbers.len() {
            let current = bucket_numbers[kk];
            if *bucket_num == current {
                found_indices.push(kk);
                count += 1;
                if count == multiplicity {
                    break;
                }
            }
        }
    }

    // TODO: add backfilling:

    assert!(found_indices.len() <= target_buckets.len());
    found_indices
}

fn lambda_to_k(lambda: f64) -> usize {
    // convert lambda to k assuming epsilon = (2-lambda)/(2*lambda)
    let k_mid = (lambda + 2.0) / (2.0 - lambda);
    let mut attempt_k = k_mid.ceil() as usize;
    if attempt_k % 2 == 0 {
        attempt_k = attempt_k + 1
    }
    attempt_k as usize
}

fn length_to_k(length: usize) -> usize {
    // get a workable k value by square rooting the number of jobs in the queue.
    let square_root = length.isqrt();
    square_root + ((square_root + 1) % 2)
}

fn queue_indices(vec: &Vec<Job>, num_servers: usize, policy: Policy, lambda: f64) -> Vec<usize> {
    // use various policies to get a vector of indices of jobs in the queue that can be worked on.
    let l_lim = 0.0;
    let u_lim = num_servers as f64;
    match policy {
        Policy::FCFS => take_to_vec(qscan(vec, num_servers)),
        Policy::PLCFS => take_to_vec(qscan(vec, num_servers)),
        Policy::SRPT => take_to_vec(qscan(vec, num_servers)),
        Policy::FCFSB => backfill(vec, num_servers),
        Policy::SRPTB => backfill(vec, num_servers),
        Policy::PLCFSB => backfill(vec, num_servers),
        Policy::LSF => take_to_vec(qscan(vec, num_servers)),
        Policy::MSF => take_to_vec(qscan(vec, num_servers)),
        Policy::LSFB => backfill(vec, num_servers),
        Policy::MSFB => backfill(vec, num_servers),
        Policy::SRA => take_to_vec(qscan(vec, num_servers)),
        Policy::LRA => take_to_vec(qscan(vec, num_servers)),
        Policy::SRAB => backfill(vec, num_servers),
        Policy::LRAB => backfill(vec, num_servers),
        Policy::DB(k) => eval_buckets(vec, k, u_lim, l_lim, false),
        Policy::DBB(k) => eval_buckets(vec, k, u_lim, l_lim, true),
        Policy::DBE => eval_buckets(vec, lambda_to_k(lambda), u_lim, l_lim, false),
        Policy::DBEB => eval_buckets(vec, lambda_to_k(lambda), u_lim, l_lim, true),
        Policy::BPT(k) => p2_buckets(vec, k, false),
        Policy::AdaptiveDoubleBucket => {
            eval_buckets(vec, length_to_k(vec.len()), u_lim, l_lim, false)
        },
        Policy::IPB(k) => {
            let set_mul_vec = k_to_partitions_mults(k);
            vec_mult_to_work(vec, k, set_mul_vec, false) 
        },
    }
}

fn simulate(
    // main simulation loop.
    policy: Policy,
    num_servers: usize,
    num_jobs: u64,
    dist: Dist,
    arr_lambda: f64,
    seed: u64,
    req_dist: Dist,
) -> f64 {
    let mut num_completions = 0;
    let mut queue: Vec<Job> = vec![];
    let mut total_response = 0.0;
    let mut time = 0.0;
    let mut rng = StdRng::seed_from_u64(seed);
    let arrival_dist = Exp::new(arr_lambda).unwrap();
    let mut total_work = 0.0;
    let mut num_arrivals = 0;

    // predict what outcome should be (if fcfs):
    if DEBUG {
        fcfstest(arr_lambda, &dist);
    }

    // initialize a first job arrival
    let mut next_arrival_time = arrival_dist.sample(&mut rng);

    while num_completions < num_jobs {
        queue.sort_by_key(|job| n64(policy.index(job)));
        if queue.len() > num_jobs.isqrt() as usize {
            println!("Error: queue length past threshold");
            break;
        }
        if DEBUG {
            println!(
                "Time is {}: | Queue: {:?} | Current work: {} Total work: {}",
                time,
                queue,
                queue.iter().map(|job| job.rem_size).sum::<f64>(),
                total_work,
            );
            std::io::stdin()
                .read_line(&mut String::new())
                .expect("whatever");
            // find next event (arrival or completion)
            // next_completion is NOT a time, it is a duration
        }

        // determine how many jobs need to get worked on in the sorted queue.
        //let num_workable = qscan(&queue, num_servers);
        //
        let mut index_workable = queue_indices(&queue, num_servers, policy, arr_lambda);
        index_workable.sort();

        if DEBUG {
            println!("Indices of jobs chosen for work: {:?}", index_workable);
        }

        let capacity: f64 = index_workable
            .iter()
            .map(|index| queue[*index].service_req)
            .sum();
        assert!(capacity < 1.0 + EPSILON);

        let next_completion = index_workable
            .iter()
            .map(|index| queue[*index].rem_size)
            .min_by_key(|f| n64(*f))
            .unwrap_or(INFINITY);

        //find next completion time out of eligible jobs
        /*
        let next_completion = queue
            .iter()
            .take(num_workable)
            .map(|job| job.rem_size as f64)
            .min_by_key(|f| n64(*f))
            .unwrap_or(INFINITY);
        */
        let timestep = next_completion.min(next_arrival_time - time);
        let was_arrival = timestep < next_completion;

        // time moves forward
        time += timestep;

        // all jobs currently in service get worked on
        /*
        queue
            .iter_mut()
            .take(num_workable) // or just 1 for now
            .for_each(|job| job.rem_size -= timestep as f64);
        */

        index_workable
            .iter()
            .for_each(|index| queue[*index].rem_size -= timestep);

        for &index in index_workable.iter().rev() {
            assert!(index < queue.len());
            if queue[index].rem_size < EPSILON {
                let job = queue.remove(index);
                total_response += time - job.arrival_time;
                num_completions += 1;
            }
        }
        /*
        for i in (0.. .min(queue.len())).rev() {
            if queue[i].rem_size < EPSILON {
                let job = queue.remove(i);
                total_response += time - job.arrival_time;
                num_completions += 1;
            }
        }
        */
        // if the job was an arrival, tick up the total work in the queue (sum of rem_sizes)
        // and add a new job to the queue.

        if was_arrival {
            total_work += queue.iter().map(|job| job.rem_size).sum::<f64>();
            num_arrivals += 1;
            let new_job_size = dist.sample(&mut rng);
            let new_service_req = req_dist.sample(&mut rng);
            let new_job = Job {
                rem_size: new_job_size,
                original_size: new_job_size,
                arrival_time: time,
                service_req: new_service_req,
            };
            queue.push(new_job);
            next_arrival_time = time + arrival_dist.sample(&mut rng);
        }
    }

    // report mean queue load
    //total_work / num_arrivals as f64
    //OR report mean response time
    total_response / num_arrivals as f64
}
