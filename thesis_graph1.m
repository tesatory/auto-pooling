y = [2.1500    2.3400    2.5700    2.9300    3.9500  5.29 6.10 6.05;
    1.6500    1.6500    1.6500    1.6500    1.6500  1.65 1.65 1.65];

x = [0 1 2 4 8 16 32 64];


plot(x,y(1,:),'ro--',x,y(2,:),'b--')
axis([0 64 0 7])
set(gca,'fontsize',14);
xlabel('Lambda')
ylabel('Temporal Coherence Score')
legend('pooled features', 'raw features', 'Location', 'NorthWest')
print -depsc2 'tc_score.eps'