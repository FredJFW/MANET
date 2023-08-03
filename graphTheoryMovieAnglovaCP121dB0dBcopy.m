%% graphTheoryMovieAnglovaCP121dB0dB.m
%% graphTheoryAnglovaMovieAnglovaCP1-121dB+-0dB
% try again 2022.01.27 ******* (WORKS GREAT) generate a movie
% from  graphTheoryAnglovaMovie.m
% from graphTheoryAnglovaFading.m
% JFW 9.4.18 clean 2020.3.7  add comments to run anglovaCP1-121dB+-0dB
% add comments so it will be easy for YM and me to "make a movie with
% "5 sec re-routing" ... explained later
% PS: to Yann. When you change this code as I would like you to do
% You have to add comments for me just like the one I am adding
%% So currently ...
%    scenario-MaxPathloss(system gain)+-fading
% scenarioCVS-plMAXallowed            +-deltadB (variable name)
% Movie for timestamp  its=[1:199 200:10:1000]

% PredicTAKEMcode (changed from PredicTAKE within MATLAB)

%
% History of graphTheoryAnglovaFading.m
% JFW 28.1.19 + 10.2.19 Dijkstra + 15.2.19 Read ext(ra) col containing
% JFW 2.4.19 added DATtot version1 DATtot=number of sec. from i->j for all
% i-j assuming it takes 1sec. to go from one node to another. see %hop%
% JFW ...8.4.19 plot i-j path with largest pathloss <= PLDiscon
% and compute max fully mesh traffic capacity M = D/??? with D/(n*(n-1))
% for fully meshed: NAN if disconnected network (for now)
%%
% Bullington's PL from Dr. Alain Jaquier, a 12th column is added
%% Fading
% JFW 10.3.18
% https://www.researchgate.net/post/How_to_calculate_Power_of_Rayleigh_Channel_Rayleigh_distributed_signal_multipath_effect
% https://www.researchgate.net/post/How_to_best_simulate_a_multipath_Rayleigh_fading_channel_using_Matlab
% You can generate a Rayleigh channel (10 taps FIR) using the following code. Remember, randn is used for Gaussian distribution (random values folllow Gaussian), when two Gaussian functions are added, the amplitude follow Rayleigh distribution.
% taps=10;
% h=randn(1,taps)+j*randn(1,taps);
% http://www.dsplog.com/2008/07/14/rayleigh-multipath-channel/


%% recall MATLAB graph theory
% G = graph([1 1], [2 3]);
% e = G.Edges
% G = addedge(G,2,3)
% G = addnode(G,4)
% or
% s = [1 1 1 2 2 3 3 4 5 5 6 7];
% t = [2 4 8 3 7 4 6 5 6 8 7 8];
% weights = [10 10 1 10 1 10 1 1 12 12 12 12];
% names = {'A' 'B' 'C' 'D' 'E' 'F' 'G' 'H'};
% G = graph(s,t,weights,names)
% plot(G)

%% for test
%load('testData/300MHzVerticalHorizLR.mat')   %use \ for PC ????
%load('anglovaLR3mAndOriginal.mat') % or csv
tic
%% mandatory lines
set(gcf,'color','white')
set(gca,'FontSize',14)
% hGiveSomeMeaningfulTitleToTheHandleOfThePlot_must start with h = figure %


readCSVData = true;
%%
if(1) %%PUT 1 to read scenario%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (readCSVData)
    clear all; close all; clc;
   % path('/Users/EIFRPOE00406/switchdrive/Armasuisse/PredicTAKE/'); DO NOT
   % USE seems to produce a BUG and then restart my MAC and/or having to
   % type restoredefaultpath
   % !!!!!!!!!
    % seems to BUg...
    readCSVData = true;
%    cd('/Users/EIFRPOE00406/switchdrive/Armasuisse/PredicTAKE/Matlab/PredicTAKE/');
    cd('/Users/EIFRPOE00406/switchdrive/Armasuisse/PredicTAKE/Matlab/PredicTAKEMcode/');
    %whitebackground% set(gcf,'color','white') %removed since I used  set(gcf,'color','white') 
    %set(gcf,'color','white') 
    readCSVData=true;    
    %scenarioCVS = 'cp1LR20.csv';
    %scenarioCVS = '1kmchainLR.csv'
    %scenarioCVS = '1kmchain55.csv'
    scenarioCVS = 'anglovaCP1.csv'
    %scenarioCVS = 'anglova55.csv'
    %scenarioCVS = 'anglovaLR3m.csv'
    %scenarioCVS = 'anglovaLR2m.csv' % was CP1LR20.csv before 10.2.19
    %scenarioCVS = '1kmchainLR_ext.csv';  %12% is missing
    %scenarioCVS = 'anglovaCP1_ext.csv';
    %scenarioCVS ='anglovaDemo.csv';
end
plMAXallowed = 121; %*******125; %121; %120  ;% 127 dB path loss max for connection

deltadB = 0.0;% dB;

axisLimits = [7.95 8 58.17 58.215]; %geo view 2,2
ax4limits=[0 25 0 90]; %stem of total hop count per nodes No 1 to 24 (by observation max 80)

%% file admin
[folder, caseName, extension] = fileparts(scenarioCVS);
caseTitle = strcat(caseName,'-',num2str(plMAXallowed),'dB');
    if(readCSVData)
      tablePL = readtable(strcat(folder,scenarioCVS));
      readCSVData = false;
    end;
figName = strcat(caseTitle,'+-',num2str(deltadB),'dB'); %UTF8 C2B1; unicode U+00B1 +-

ForMovieFolder = strcat('movie/',figName);
[status, msg, msgID] = mkdir(ForMovieFolder)
addpath(ForMovieFolder)  
%savepath /Users/EIFRPOE00406/switchdrive/Armasuisse/PredicTAKE/Matlab/pathdef.m

%    specifier = datestr(now,'HH_MM_SS'));
specifier = '';

%% verify unless your are sure
scatter(tablePL.d,tablePL.pathloss)
scatter3(tablePL.ts,tablePL.d,tablePL.pathloss)

%% ALL timestamp analysis
pl = tablePL.pathloss; 
%pl_b = tablePL.pathloss_b; %b for Bullington's model 
pl_b = tablePL.pathloss; %b for scenarioCVS = '1kmchain55.csv'

indexPL =  pl_b >= plMAXallowed-deltadB & pl_b <= plMAXallowed+deltadB;


% NOT USED titleCaptionWithij =strcat(figName,'(i=blue, j=red)');

% TRICK TO GET ALL NODES i and j coordinates.
% FASTER THEN "Bts = vertcat(Btsij,Btsji);" TRICK ? (SEE IN THE LOOP BELOW)
%     uniqueI=unique([tablePL.i_nem_id tablePL.i_lon tablePL.i_lat],'rows');
%     uniqueJ=unique([tablePL.j_nem_id tablePL.j_lon tablePL.j_lat],'rows');
%     uniqueNodes = unique([uniqueI;uniqueJ],'rows');  %[uniqueI;uniqueJ] = vertical concatenation due to ;
%     pXData = uniqueNodes(:,2);
%     pYData = uniqueNodes(:,3); 
    % verification
%   nodeListe = unique(uniqueNodes(:,1));
%     scatter(pXData,pYData)
%     axis([7.95 8.00 58.17 58.215])
%     axis('equal')
%     grid

%% FIGURES for analysis of the whole figure
% interesting but hard to conclude anything ... too much data
%REMOVED from graphTheoryAnglovaFading.m
%if(0) %test figure
%...
% keep only lon,lat,pathloss view from i BUT view from j WAS MISSING
% (8.4.19)
%end%% if(1) test figure

%% comparison of pathloss 
%REMOVED from graphTheoryAnglovaFading.m

%% 
end%% if(1) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% sort according to ts then i then pathloss
%    ts       d       i_nem_id    i_lat    i_lon     i_alt    j_nem_id    j_lat     j_lon     j_alt    pathloss
tablePLSorted = sortrows(tablePL,[1,3,11]);

%keep pl or set
%pl = tablePL.pathloss_b + 11;
addToTitle = ' ';
%addToTitle = 'B+11dB';
%addToTitle = 'Holm'; for future use

ts = tablePLSorted.ts; %company1full.ts; 


indexPllessthanPmin = pl <= plMAXallowed; %Pmin <=> Pmax
% nNEMs = max(tablePLSorted.j_nem_id); is this USED ???? JFW 6.8.19
cp1OneHop = tablePLSorted(indexPllessthanPmin,:);
% JFW 6.7.19 why did I use tablePL and not tablePLSorted ... 
% Sorting must be (re-)done later for each ts due to missing i-j
%B = sortrows(cp1OneHop,[1,3,11]); %sorted: ts first then i_nem_id then pathloss
% draw for each ts the line i-j with j such that i-j has the largest path
% loss >= plMAX, i.e., in coverage from i


numberOfts = ts(end);
numberOftsWithDisconnection = zeros(numberOfts,1);
maxSP = zeros(numberOfts,1);
minSP = zeros(numberOfts,1);
meanSP = zeros(numberOfts,1);
maxSP = zeros(numberOfts,1);
meanSPwith0 = zeros(numberOfts,1);
DATtot = zeros(numberOfts,1); %hop%
%for test 
%for its=68:68 %68 is (disconnected)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%for its=1:10:numberOfts
%for its=[1:199 200:10:1000]  or for its=[1 200 600 1000]
for its=1:1:1000
    
%for its=ts(1):1:ts(end) %1:1:1002
%DEBUG    its = 68;

    fileFigName = strcat(figName,'-',num2str(its));
    titleCaptionts = strcat('ts=',num2str(its),' ',figName);
    title(titleCaptionts); %https://www.mathworks.com/matlabcentral/answers/108652-draw-lines-between-points

indexBts = (tablePLSorted.ts == its);
Btsij = tablePLSorted(indexBts,:);
dmax = max(Btsij.d);
% FOR EACH i IN Bts find j with the largest path loss (i.e, the last
% elements)
% the problem is that some i node might be missing since only i-j are
% listed not j-i due to symmetrical link
% Brute force approach
%permute i and j, i.e., col 3456 and 78910 https://ch.mathworks.com/matlabcentral/answers/155317-reorder-table-variables-columns
Btsji = Btsij(:,[1 2 7 8 9 10 3 4 5 6 11]); %permute i and j just like this ... it works ... Matlab is great
newB = vertcat(Btsij,Btsji); %https://ch.mathworks.com/matlabcentral/answers/279758-join-two-tables-with-the-same-headers
% above does not work because MATLAB is "too" smart and concatenate the
% "correct" column according to their names.
Btsji.Properties.VariableNames = Btsij.Properties.VariableNames; %https://ch.mathworks.com/matlabcentral/answers/431337-change-variable-names-in-a-table
BtsWithDoublon = vertcat(Btsij,Btsji); %now it is what I want
% Bts = BtsForDirectedGrap;
Bts = unique(BtsWithDoublon,'stable'); %stable = same order as in original, i.e., i_NEM_ID, j_NEM_ID, pathloss

% Bts contains all i-j as directed graph (i.e., j-i is repeated)
% i.e., Bts.Properties.VariableNames Columns 1 through 11 (or 12)
%{'ts'}    {'d'}    {'i_nem_id'}    {'i_lat'}    {'i_lon'}    {'i_alt'}    {'j_nem_id'}    {'j_lat'}    {'j_lon'}    {'j_alt'}    {'pathloss'}

% trash the following ******************************
%col_i_nem_id = Bts.i_nem_id;
% use unique since may be i-j and j-i are represented twice
%[list_i_nem_id,ia,ic] = unique(col_i_nem_id,'last'); %[C, ia, ic] = unique(A)
% list_i_nem_id = col_i_nem_id(ia) = 
% **********************************************
%xunique=[Bts.i_lon(ia) Bts.j_lon(ia)]; to plot only ONE out of all possible link
%yunique=[Bts.i_lat(ia) Bts.j_lat(ia)];

%x=[Bts.i_lon Bts.j_lon]; %to plot all possible link
%y=[Bts.i_lat Bts.j_lat];

indexPathlossOK = Bts.pathloss <= a<y;

x=[Bts.i_lon(indexPathlossOK) Bts.j_lon(indexPathlossOK)]; %to plot all possible link
y=[Bts.i_lat(indexPathlossOK) Bts.j_lat(indexPathlossOK)];

[C, index_unique_nem_id, indexC] = unique(Bts.i_nem_id);
pXData = Bts.i_lon(index_unique_nem_id);
pYData = Bts.i_lat(index_unique_nem_id);
%verification
%scatter(pXData,pYData)

    % Enlarge figure to full screen. %[x y width height], where x and y define the distance from the lower-left corner of the screen to the lower-left corner of the figure
    % https://ch.mathworks.com/matlabcentral/answers/173629-how-to-change-figure-size
    
%sub(1,3,2)  %%%%%%%%%%%%%%%%%%%%% largest connected PL ************
%    plot(x',y')
%    axis([7.95 8 58.17 58.215])
%    xlabel('lon')
%    ylabel('lat')
%    axis('equal')
    %plot_google_map('MapType','satellite')
    %pbaspect([1 1 1]) does not work here %axis(square) never work

    % verification
    %figure
    %scatter(Bts.i_nem_id(ia),Bts.pathloss(ia)) 
    Sg = Bts.i_nem_id(indexPathlossOK); %Source Node in graph theory jargon
    Tg = Bts.j_nem_id(indexPathlossOK); %
    Wg = Bts.pathloss(indexPathlossOK);
    GpathlossWithDoublon = graph(Sg,Tg,Wg);
    
    %figure; histogram(Sg)
    %figure; histogram(Tg)
%    Gpathloss_Edges = Gpathloss.Edges;
    Gpathloss = simplify(GpathlossWithDoublon);
    Apathloss = adjacency(Gpathloss,'weighted');
    Apathloss_display = full(Apathloss);
    Apathloss_display(Apathloss_display==0)=NaN;


    
%     ABinary = adjacency(Gpathloss);
%     ABinary_display = full(ABinary);
%     GBinary = graph(ABinary);

    WgBinary = double(Wg > 0);
    GBinary = graph(Sg,Tg,WgBinary);
    %plot(Gtemp);
    ABinary = adjacency(GBinary); % = adjacency(GBinary)
    %spy(Atemp);
    hopCountij = distances(GBinary);
    hopCountij(isinf(hopCountij)) = NaN; %infinity set to NaN
    sumColPerRow = sum(hopCountij,'omitnan');
    % stem(sumColPerRow) in subplot 2,2
    totalNumberOfHops = sum(sumColPerRow); %366 for ts=68
    
    DoMOptimalScheduling = totalNumberOfHops;
    
    % COMPUTE DoM ideal routing RR
    numberOfNodes = height(GBinary.Nodes); %or = length(unique([Sg' Tg']));
    DoMIdealRoutingRR = numberOfNodes*max(sumColPerRow);
    
%% COMPUTE  nHops = average number of hops (over >0 hops)
    hopCountij0NAN = hopCountij;
    hopCountij0NAN(isnan(hopCountij0NAN))=0;
    nonZeroElementsInHopCountij = nonzeros(hopCountij0NAN);
    numberNonZero = length(nonZeroElementsInHopCountij) 
    sum(nonZeroElementsInHopCountij)/ numberNonZero %same as
    
    
    
    nHops = mean(nonZeroElementsInHopCountij); %average number of hopCountij entries >0   
   %nRouting = nHops-1;  
   %LQ = 1;
   %NLQ = 1;
   %ETX = 1./LQ./NLQ./WgBinary
   %DAT = 2./LQ./POR
   
   %%START ANALYSI%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %% Analysis
%    Atemp = adjacency(Gpathloss);
%    Aunity = adjacency(Gunity);
%    AtempW = adjacency(Gpathloss,'weighted');

%     figure
%     hSg=histogram(Sg)
%     figure
%     hTg=histogram(Tg)
%     figure; title('adjacency(graph(Sg,Tg,pathloss))')
%     spy(Atemp)
%     
%     figure; title('adjacency(graph(Sg,Tg))')
%     spy(Gpathloss)
%     figure
%     
%     plot(Atemp)
%     ax =gca;
%     ax.XAxis.MinorTick = 'on'
%     ax.YAxis.MinorTick = 'on'
%     grid on
%     grid minor
%     
%     spy(Atemp) %plots the sparsity pattern of the sparse adjacency matrix A.
    

binsG = conncomp(Gpathloss); %if G was a digraph then you would need 'Type','weak'); %weak_bins for undirected graph (digraph=directed graph) 
 isConnected = all(binsG == 1);
 
numberOftsWithDisconnection(its) = isConnected;

 if(isConnected)
     textConnectedOrNot = 'connected';
 else
      numberOftsWithDisconnection(its) = 1;
      textConnectedOrNot = 'disconnected'; 
 end    

if(0)
        %hop% find OPTIMAL SCHEDULE
        dtot=0;
        % verifiy numberOfNodes
        Nnodes = numnodes(G); %https://ch.mathworks.com/help/matlab/ref/graph.numnodes.html
        for i=1:Nnodes
        for j=1:Nnodes
            [Pij,dij] = shortestpath(G,i,j); %
            %ideal schedule = schedule according to path given by Pij
            %Example: [Pij,dij] = shortestpath(G,1,7) gives 1    16     3     7
            %and dij 3
            %ideal schedule = 1    16     3
            dtot = dtot+dij;
        end
        end
        DATtot(its) = dtot; %gives 1274 for its=1 for anglovaCP1.csv and 1118 for its=68 for 1kmchain55.csv
        %dtot is the time it takes for node i to be able to send an other message to node j
end

%% M values (computation)
%Assuming
Dmax = 1e6; %b/s
% Assume a slot duration 
% MessageSizeB = 100; %B
% T=MessageSizeB*8/Dmax;
% Ttot = T*DATtot(its);
% % M = MessageSizeB*8/Ttot = T*D/(T*dtot)) =  D/dtot;
% M = MessageSizeB*8/Ttot; %b/s or
% M = Dmax/DATtot(its); %max bit rate for each nodes to each other nodes on graph G
% NMessagePerSec = M/(MessageSizeB*8);%b/s/(b/message)=message/s per node = Ttot 
% Max bit rate for i-j equi-distribution traffic M = 1/DATtot(its)*D
% if D= 1Mb/s => M= 785 kb/s = 98 kB/s
% if msgNB=200 [B] data => NmessagePerSec = 98e3/200 = 490 messages/s

% discussion with Gilles:
%24 noeuds 1002s scenario 24048 message => 24048/1002s/24 = 24 messages/s/24node= 1 message/s per node to anyother node 
%every seconds 23 nodes sends x messages to another nodes 

DoM_FullyMeshed = numberOfNodes^2 - numberOfNodes; %=24*23=552
DoM_tInLine = 1/3*(numberOfNodes^3 - numberOfNodes); %=24*23=552

MinLine_kpbs = round(Dmax/DoM_tInLine/1000,1); %kbps
MRR_kpbs = round(Dmax/DoMIdealRoutingRR/1000,1); %kpbs                                                                                                                                                                                                                                                = d/DoMOptimalScheduling/1000; %kpbs %= sum(hopCountij,'all')
Mopti_kpbs = round(Dmax/DoMOptimalScheduling/1000,1); %kpbs %= sum(hopCountij,'all')
MFullyM_kpbs = round(Dmax/DoM_FullyMeshed/1000,1); %kpbs
    
    
    
%% Mvalues string
% = strcat('IL=',num2str(MinLine),'<RR=',num2str(MRR),'<OP=',num2str(Mopti),'<FM=',num2str(MFullyM),'<<D',num2str(round(Dmax/1000)),'[kpbs]');
Mvalues1 = strcat('IL < RR < OP < FM << D  (CR=100%)');
Mvalues2 = strcat(num2str(MinLine_kpbs),'<',num2str(MRR_kpbs),' < ',num2str(Mopti_kpbs),' < ',num2str(MFullyM_kpbs),' << ',num2str(round(Dmax/1000)),'[kpbs]');
Mvalues3 = strcat(num2str(DoM_tInLine),'>',num2str(DoMIdealRoutingRR),'>',num2str(DoMOptimalScheduling),'>',num2str(DoM_FullyMeshed),'>> 1 [slot]');

Mvalues = {Mvalues1;Mvalues2;Mvalues3}


%% CR       
   routeCountij = max(0,hopCountij-1);
   routeCount = sum(routeCountij,'all')/(numberOfNodes^2-numberOfNodes);

    CRij= hopCountij; CRij(CRij > 0) = 1; CRtot = sum(sum(CRij,'omitnan'));
    
    CR = round(100*CRtot/(numberOfNodes^2-numberOfNodes))
    
%% RTT    
    %RTT = Nb/M + CR/100*Tproc + routeCount*Troute
    
    nB=100; %B
    nb=nB*8; %bit    
    RTTD = 2*nb/Dmax; %s over 2 nodes only no RR
    latencyRR = 2*nb/(Dmax / numberOfNodes); %s over 2 nodes using RR

                 Mopti = Dmax / DoMOptimalScheduling ; %bps 
    RTTM = 2*nb/Mopti; %s

                  MRR   = Dmax / DoMIdealRoutingRR ; %bps %%%%%%%%%%%%%%
    RTTRR = 2*nb/MRR; %s

    Tproc = 1e-3; %s
    Troute = 10e-3; %s

    RTTestimation = 2*((latencyRR + Tproc)*nHops + Troute * (nHops-1)); %s  %    (nHops-1) = Nrouting
    xlabelGeoView1 = 'lon';
    xlabelGeoView2 = strcat('RTTD=',num2str(RTTD),', RTTM=',num2str(RTTM),', RTTRR=',num2str(RTTRR),'[s]');
    xlabelGeoView3 = strcat('RTT=2((',num2str(latencyRR),'+',num2str(Tproc),')*',num2str(nHops),'+',num2str(Troute),'*(',num2str(nHops-1),')');
    xlabelGeoView = {xlabelGeoView1,xlabelGeoView2,xlabelGeoView3}

%% RTT details
%     NbOverDs = nB*8/Dmax; %s
%     NbOverMRRs = nB*8/MRR;%s
%     NbOver
%     RTToptimalSchedulems = round(NbOverDs*1000);
%     RTTidealRRms = round(NbOverMRRs*1000);
%     label1 = strcat('100B/D: optimal=',num2str(RTToptimalSchedulems),' or RR=',num2str(RTTidealRRms));
%     label2 = strcat('+',num2str(AverageNprocessing),'*Tproc+',num2str(AverageNrouting),'*Tproc');
%     RTTtext = {label1;label2}; %cell array see https://ch.mathworks.com/help/matlab/ref/title.html#input_argument_d0e1024746  

    
    %% Disconnected graph or not
    GPl=Gpathloss;
    binsG = conncomp(GPl); %if G was a digraph then you would need 'Type','weak'); %weak_bins for undirected graph (digraph=directed graph) 
    isConnected = all(binsG == 1);
    if(isConnected)
        textConnectedOrNot = 'conn.';
    else
        textConnectedOrNot = 'disc.'; 
    end
    
    
    
    
    [minTotHop, nodeMinHop] = min(sumColPerRow); %https://stackoverflow.com/questions/13531009/how-can-i-find-the-maximum-value-and-its-index-in-array-in-matlab
    [maxTotHop, nodeMaxHop] = max(sumColPerRow); %https://stackoverflow.com/questions/13531009/how-can-i-find-the-maximum-value-and-its-index-in-array-in-matlab
    label_sumColPerRow1 = 'Node number';
    label_sumColPerRow2 =strcat('min=',num2str(minTotHop),'@node',num2str(nodeMinHop(1)), '. MAX=',num2str(maxTotHop),'@node',num2str(nodeMaxHop(1)));    
%    label_sumColPerRow2 =strcat('min hop=',num2str(minhop),' for ',num2str(MinHopNodesNbr), 'nodes. MAX hop=',num2str(maxhop),' for ',num2str(MaxHopNodesNbr),'nodes');
    label_sumColPerRow = {label_sumColPerRow1,label_sumColPerRow2}
    
%% Number of node at 1 and 2 hops from hopCountij
NumberOfNodeat1hop = sum(hopCountij(:,:)==1); %https://ch.mathworks.com/matlabcentral/answers/142281-count-the-number-of-times-a-value-occurs-in-a-specific-of-an-array
NumberOfNodeat2hop = sum(hopCountij(:,:)==2); %https://ch.mathworks.com/matlabcentral/answers/142281-count-the-number-of-times-a-value-occurs-in-a-specific-of-an-array
PercentOfNodeat1hop = NumberOfNodeat1hop/(numberOfNodes-1);
PercentOfNodeat2hop = NumberOfNodeat2hop/(numberOfNodes-1);

    
%% title case ts CR RTT
titleCaseTsCRRTT= strcat('Graph view: ts=',num2str(its),', ',textConnectedOrNot,', CR=',num2str(CR),'%',' RTT=',num2str(round(RTTestimation,3)),'s');
%% END ANALYSIS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% PLOTS    
% 11/16 if 4 subplots
%set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0, 0.8*11/16, 0.8]);%[x y width height] ...https://ch.mathworks.com/matlabcentral/answers/173629-how-to-change-figure-size
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0, 1, 1]);%[x y width height] ...https://ch.mathworks.com/matlabcentral/answers/173629-how-to-change-figure-size

subplot(2,3,1) %%%%%%%%%%%%%%%%%%%%% GRAPH VIEW **********************
    % p=plot(G,'EdgeLabel',round(G.Edges.Weight))
    LWidths = (round(max(GPl.Edges.Weight))./round(GPl.Edges.Weight)).^4; %empirical trick to make "it looks good", i.e., larger =>better (lower path loss)
    p=plot(GPl,'EdgeLabel',round(GPl.Edges.Weight),'LineWidth',LWidths);

    title(titleCaseTsCRRTT)
    %printCurrentFigure;
    %     set(gcf,'PaperPositionMode','auto')
    %     fileName = strcat(num2str(its),'-D-',caseTitle,datestr(now,'HH_MM_SS'));
    xlabel( Mvalues )

subplot(2,3,2) %%%%%%%%%%%%%%%%%%%%% Geographical VIEW **********************
    p=plot(GPl,'EdgeLabel',round(GPl.Edges.Weight),'LineWidth',LWidths);
    title(strcat('Geo. view: ts =',num2str(its),' dmax=',num2str(round(dmax,3)),'km')); 
    
    p.XData = pXData; %put it back in the table
    p.YData = pYData;

    axis(axisLimits)
    xlabel(xlabelGeoView)
    ylabel('lat')
%    axis('equal')
    
    if (its ==1)
          plot_google_map('satellite')
          axis(axisLimits)  %NOT SURE IF this is REALLY NEEDED
    end
    hold off % if you do hold off here then ... 
    
%      subplot(1,3,3)
%  %    p=plot(G,'EdgeLabel',round(G.Edges.Weight))
%     LWidths = (round(max(GPl.Edges.Weight))./round(GPl.Edges.Weight)).^4;
%     p=plot(GPl,'EdgeLabel',round(GPl.Edges.Weight),'LineWidth',LWidths)

subplot(2,3,3)  %%%%%%%%%%%%%%%%%%%%% number of one and two hops per node ************    

plot(PercentOfNodeat1hop*100,'bo'); hold on
plot(PercentOfNodeat2hop*100,'r*'); hold on
plot((PercentOfNodeat1hop+PercentOfNodeat2hop)*100,'k-s','MarkerEdgeColor','k'); hold off
legend('1 hop','2 hops','1&2 hops')
set(legend,'color','none','Location','southwest'); %https://ch.mathworks.com/matlabcentral/answers/97850-how-do-i-make-the-background-of-a-plot-legend-transparent
grid
axisPercentNumberOfHops = [0 25 0 100]
axis(axisPercentNumberOfHops)
title(' % of nodes at one and 2 hops')
    

subplot(2,3,4)  %%%%%%%%%%%%%%%%%%%%% Adj ************    
%    imagesc(-Apathloss_display,[-170 -50])
%    heatmap(-Apathloss_display,[-170 -50])DOES NOT WORK AS IN HELP
    hmapFigure=heatmap(Apathloss_display); % look nice !
    hmapFigure.ColorLimits = [50 170];
%    pbaspect([1 1 1])   
    colormap(jet)
    colorbar
    title('Adjacency Matrix of pathloss')
    
subplot(2,3,5)  %%%%%%%%%%%%%%%%%%%%% Adj ************    
    hmapFigure=heatmap(hopCountij); % look nice !
    hmapFigure.ColorLimits = [0 7];
%    pbaspect([1 1 1])   get ERROR Error using matlab.graphics.chart.HeatmapChart/set
%The name 'plotboxaspectratio' is not an accessible property for an
%instance of class 'matlab.graphics.chart.HeatmapChart'.

    colormap(flipud(jet)) %colormap(flipud(hot))
    colorbar
    title('Adjacency Matrix of hop count')
    

ax4 = subplot(2,3,6)  %%%%%%%%%%%%%%%%%%%%% Adj ************    
title(ax4, 'Total Number of hops to all j')
stem(ax4, sumColPerRow)
xlabel(label_sumColPerRow)
ylabel('Total Number of hops')
grid
axis(ax4, ax4limits)
hold off
    
%
%xlabel(....);
%legend('Max hops', 'Average hops', 'Min hops','Avg. hops (without no-hop)','Reachable Nodes', 'Location','northeast') % victor 3.1.19    
%grid
%set(hFig,'Position',[440 460 560 340])


    
%% PRINT FIGURE EVERY TS FOR MOVIE    
%       printCurrentFigure;
set(gcf,'PaperPositionMode','auto')
%    fileName = strcat(num2str(its),caseTitle,datestr(now,'HH_MM_SS'));
    fileName = strcat(num2str(its),'-',caseTitle,specifier);
% pause(0.5);
    print(strcat(ForMovieFolder,'/',datestr(now,'HH_MM_SS'),'-',fileFigName), '-dpng','-r0');% 
    pause(1);
    close all
    
 end %if(0)
toc
tic
% then run MakeAMovieFromPNGinFolder.m 
imageFolder2mpeg(ForMovieFolder)

toc