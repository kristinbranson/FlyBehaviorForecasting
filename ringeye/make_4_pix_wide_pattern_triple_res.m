% make 4 wide stripe pattern for simulation, this pattern uses 3 frames for
% a smoother expansion signal

InitPat = [repmat([-ones(1,3*4), ones(1,3*4)], 1,6), repmat([ones(1,3*4), -ones(1,3*4)], 1,6)];
temp_Pats(:,:,1) = InitPat;

for j = 2:24
    temp_Pats(:,:,j) = simple_expansion(temp_Pats(:,:,j - 1), 145);    %centered expansion, from 288/2 + 1    
end

% make first set of expansion pats
for j = 1:24
    for i = 1:96
        Pats(:,i,1,j) = (sum(temp_Pats(:,((i*3)-2):i*3,j))./3);
    end
end    

for j = 1:24
    for i = 2:96
        Pats(:,:,i,j) = ShiftMatrix(Pats(:,:,i-1,j), 1, 'r', 'y'); 
    end
end

% for j = 1:24   % lazy way, but shifts pattern so it is centered...
%     for i = 1:96
%         Pats(:,:,i,j) = ShiftMatrix(Pats(:,:,i,j), 3, 'l', 'y'); 
%     end
% end

% 
% %plot for debugging
% for i = 1:96
%     for j = 1:24
%         imagesc(squeeze(Pats(:,:,i,j)))
%         pause(0.5)
%     end
% end


