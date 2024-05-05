function [Basari]=Wisconsin_Costile_1fold(w_rel,w_cost)
% 2 FEATURES
% TRAIN Verisi
load('C:\Users\suheyla\Desktop\TÝK444\Wisconsin\Wisconsin.mat')
% Cikis uyelik fonk.
hasta_bilgisi=0:1; %0 Hasta,  1 Saglikli
u_hasta=trimf(hasta_bilgisi,[0 0 0]);
u_saglik=trimf(hasta_bilgisi,[1 1 1]);

load('C:\Users\suheyla\Desktop\TÝK444\Wisconsin\A1.mat')
test_data=A1(:,1:9);
test_result=A1(:,10); % 0 for hasta, 1 for saglik

%PROBABILITIES
w_rel=w_rel; % Cost1(Rel) ve Cost2(Ent.)'nin aðýrlýklarý.
w_cost=w_cost; % Cost1'in agirligi
% x1_psmall=[0.8761;0.7795;0.7021;0.6387;0.5858;0.5410;0.5025;0.4692;0.44;0.4142];
% x1_plarge=1-x1_psmall;
% [u_x1_Small,u_x1_Large]=wiscon_mem_atayici(w_rel,w_cost,psmall,plarge);

x3_psmall=[0.8157;0.6887;0.596;0.5252;0.4695;0.4245;0.3873;0.3561;0.3296;0.3068]';
x3_plarge=1-x3_psmall;
[u_x3_Small,u_x3_Large]=wiscon_mem_atayici(w_rel,w_cost,x3_psmall,x3_plarge);

x8_psmall=[0.7872;0.6491;0.5522;0.4805;0.4253;0.3814;0.3458;0.3162;0.2913;0.2700]';
x8_plarge=1-x8_psmall;
[u_x8_Small,u_x8_Large]=wiscon_mem_atayici(w_rel,w_cost,x8_psmall,x8_plarge);

%RANGE FINDER
[x3_range_small,x3_range_large]=wiscon_rangefinder(u_x3_Small,u_x3_Large);
[x8_range_small,x8_range_large]=wiscon_rangefinder(u_x8_Small,u_x8_Large);

%OUTPUT RELIABILITY
[R1_out_rel,R2_out_rel,R3_out_rel,R4_out_rel]=wiscon_2rule_output_relfinder(x3_range_small,x3_range_large,x8_range_small,x8_range_large,All); 

%TEST
for x=1:length(test_data)
%     giris_x1=test_data(x,1);
%     giris_x2=test_data(x,2);
    giris_x3=test_data(x,3);
    giris_x8=test_data(x,8);

 % Probability of Input
 [x3_probsmall,x3_problarge]=wiscon_input_prob_finder(giris_x3,x3_psmall,x3_plarge);
 [x8_probsmall,x8_problarge]=wiscon_input_prob_finder(giris_x8,x8_psmall,x8_plarge);

 % Membership of Input
 [x3_uSMALL,x3_uLARGE]=wiscon_memb_finder(giris_x3,u_x3_Small,u_x3_Large);
 [x8_uSMALL,x8_uLARGE]=wiscon_memb_finder(giris_x8,u_x8_Small,u_x8_Large);

 % ÝÞLEMLER
% Bu Örnek için 4 kural var.
% Kurallar
K1=x3_uSMALL*x3_probsmall*x8_uSMALL*x8_probsmall*R1_out_rel;
K2=x3_uSMALL*x3_probsmall*x8_uSMALL*x8_probsmall*R2_out_rel;
K3=x3_uLARGE*x3_problarge*x8_uLARGE*x8_problarge*R3_out_rel;
K4=x3_uLARGE*x3_problarge*x8_uLARGE*x8_problarge*R4_out_rel;

Payda=K1+K2+K3+K4;
Pay=K1*u_hasta+K2*u_saglik+K3*u_hasta+K4*u_saglik;

RESULT(x,:)=Pay/Payda;
end

for i=1:length(RESULT)
    if RESULT(i,1)>RESULT(i,2)
        Y(i,1)=0;
    elseif RESULT(i,2)>=RESULT(i,1)
        Y(i,1)=1;
    else Y(i,1)=-1;
    end
end

Dogru=0; Yanlis=0;
for i=1:length(RESULT)
    if Y(i,1)==test_result(i,1)
        Dogru=Dogru+1;
    else Yanlis=Yanlis+1;
    end
end
Basari=Dogru/(Dogru+Yanlis)*100;
