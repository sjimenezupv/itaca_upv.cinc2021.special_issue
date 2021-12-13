function H=shannonEntropy(x)
    h=hist(x,200)/length(x);
    h=h(h~=0);
    H=-sum(h-log2(h));
end