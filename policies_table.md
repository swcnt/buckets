
| FCFS, | First come, first served |
| :---- | :---- |
| PLCFS, | Preemptive last come, first served |
| SRPT, | Shortest remaining processing time (lowest rem\_size) |
| FCFSB, | First come, first served \+ backfilling |
| SRPTB, | Shortest remaining processing time \+ backfilling |
| PLCFSB, | Preemptive last come, first served \+ backfilling |
| LSF, | Least servers first (lowest service\_req) |
| LSFB, | Least servers first \+ backfilling |
| MSF, | Most servers first |
| MSFB, | Most servers first \+ backfilling |
| SRA, | Smallest remaining area (rem\_size \* service\_req) |
| SRAB, | Smallest remaining area \+ backfilling |
| LRA, | Largest remaining area |
| LRAB, | Largest remaining area \+ backfilling |
| DB(usize), | Double bucket (two buckets are picked out of k total buckets (or one largest one) and one job from each bucket is worked on) |
| DBB(usize), | Double bucket \+ backfilling |
| DBE, | Double bucket, but k is chosen as a function of lambda. (k \= (lambda+2)/(2-lambda)) because epsilon \= (2-lambda)/(2*lambda) |
| DBEB, | Double bucket adapting to k \+ backfilling |
| BPT(usize), | Buckets of powers of two (eg. eight ones, two fours, if k is 8\) |
| AdaptiveDoubleBucket, | Adaptive Double Bucket (based on \# of jobs in queue) |

