%% sdLookUp.m
% JFW 2023.8.25

% usage
% sdIndex = sd(s,d);
% instead of sdIndex = s*10+d
% 


sd = nan(6);
% pair close together
sd(1,3) = 1;
sd(3,1) = 2;

sd(2,4) = 4-0.5;
sd(4,2) = 5-0.5;

sd(5,6) = 6;
sd(6,5) = 7;

% 13 - 56 ; 1&3 in C00.10; 5&6 are moved first than stay in C10.07
sd(1,5) = 10;
sd(5,1) = 11;
sd(3,5) = 12;
sd(5,3) = 13;

sd(1,6) = 15;
sd(6,1) = 16;
sd(3,6) = 17;
sd(6,3) = 18;

% 13 - 24 1&3 in C00.10; 5&6 stay in C10.07 after ?? min, 2&4 are moved
sd(1,2) = 20;
sd(2,1) = 21;
sd(3,2) = 22;
sd(2,3) = 23;

sd(1,4) = 25;
sd(4,1) = 26;
sd(3,4) = 27;
sd(4,3) = 28;

% 24 - 56         2&4 in C00.10 for x min, then 5&6 are moved first than stay in C10.07
%                       fixed pairs during the last ?? min => no change
%                       expected
sd(5,2) = 30;
sd(2,5) = 31;
sd(6,2) = 32;
sd(2,6) = 33;

sd(5,4) = 34;
sd(4,5) = 35;
sd(6,4) = 36;
sd(4,6) = 37;
