function [integerVec] = generateRandomSequence(minVal, maxVal, vecLength)

integerVec = zeros(vecLength, 1);
for i = 1:vecLength
    found = 1;
    while found == 1
        found = 0;
        val = round(unifrnd(minVal, maxVal, [1, 1]));
        for j = 1:maxVal/2
            if i > j
                if integerVec(i-j) == val
                    found = 1;
                end
            end
        end
        if found == 0
            integerVec(i) = val;
        end
    end
end