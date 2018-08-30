function Model = CyTOF_LDAtrain(SamplesFolder,mode,LabelsFolder,RelevantMarkers,arcsinh_transform)
% CyTOF_LDAtrain function can be used to train a Linear Discriminant
% Analysis (LDA) classifier, on the labeled CyTOF samples. The trained
% classifier model can be then used to automatically annotate new CyTOF
% samples.
%
% Input description
%
% SamplesFolder: extension of the folder containing the training samples,
%                can be either in FCS or csv format.
%
% mode:         either 'FCS' or 'CSV', defining the samples format.
%
% LabelsFolder: extention of the folder containing the labels (cell types)
%               for the training samples, must be in csv format. Labels
%               files must be in the same order as the samples, to match
%               each lables file to the corresponding sample.
%
% RelevantMarkers: list of integers, enumerating the markers to be used for
%                  classification.
%
% arcsinh_transform: True = apply arcsinh transformation with cofactor of 5
%                    prior to train the classifier.
%                    False = no transformation applied.
%
% Example
% Model = CyTOF_LDAtrain('HMIS-2\Samples\','CSV','HMIS-2\Labels\',[1:28],True)
%
% For citation and further information please refer to this publication:
% "Predicting cell types in single cell mass cytometry data"

% read the training data
SamplesData = struct('Data',[],'Labels',{});

if strcmp(mode,'FCS')
    H=dir(fullfile(SamplesFolder,'*.fcs'));
    SamplesFiles = cellstr(char(H(1:end).name));
    SamplesFiles = sort_nat(SamplesFiles);
    for i = 1:length(SamplesFiles)
        SamplesData(i).Data = fca_readfcs([SamplesFolder SamplesFiles{i}]);
        SamplesData(i).Data = SamplesData(i).Data(:,RelevantMarkers);
    end
    clear i
elseif strcmp(mode,'CSV')
    H=dir(fullfile(SamplesFolder,'*.csv'));
    SamplesFiles = cellstr(char(H(1:end).name));
    SamplesFiles = sort_nat(SamplesFiles);
    for i = 1:length(SamplesFiles)
        SamplesData(i).Data = csvread([SamplesFolder SamplesFiles{i}]);
        SamplesData(i).Data = SamplesData(i).Data(:,RelevantMarkers);
    end
    clear i
else
    msgbox('Invalid file format mode, choose FCS or CSV', ...
        'Error','error');
    return;
end

H=dir(fullfile(LabelsFolder,'*.csv'));
LabelsFiles = cellstr(char(H(1:end).name));
LabelsFiles = sort_nat(LabelsFiles);
for i = 1:length(LabelsFiles)
    SamplesData(i).Labels = table2cell(readtable([LabelsFolder LabelsFiles{i}],'ReadVariableNames',0,'Delimiter',','));
    if (length(SamplesData(i).Data) ~= length(SamplesData(i).Labels))
        msgbox(['Length of training data and labels of sample ' i ' does not match'], ...
            'Error','error');
        return;
    end
end
clear i H SamplesFiles LabelsFiles

Data=[];
Labels=[];
for i = 1:length(SamplesData)
    Data = [Data; SamplesData(i).Data];
    Labels = [Labels; SamplesData(i).Labels];
end
clear i SamplesData

% Apply arcsinh tranformation
if (arcsinh_transform)
    Data=asinh((Data-1)/5);
end

% Train LDA classifier
Model.LDAclassifier = fitcdiscr(Data,Labels);    
Model.arcsinh = arcsinh_transform;      % Keep information if the model works on transformed data or not
Model.markers = RelevantMarkers;        % Keep information of the markers used to train the model
end
%%
function [fcsdat, fcshdr, fcsdatscaled, fcsdatcomp] = fca_readfcs(filename)
% [fcsdat, fcshdr, fcsdatscaled] = fca_readfcs(filename);
%
% Read FCS 2.0 and FCS 3.0 type flow cytometry data file and put the list mode
% parameters to the fcsdat array with size of [NumOfPar TotalEvents].
% Some important header data are stored in the fcshdr structure:
% TotalEvents, NumOfPar, starttime, stoptime and specific info for parameters
% as name, range, bitdepth, logscale(yes-no) and number of decades.
%
% [fcsdat, fcshdr] = fca_readfcs;
% Without filename input the user can select the desired file
% using the standard open file dialog box.
%
% [fcsdat, fcshdr, fcsdatscaled] = fca_readfcs(filename);
% Supplying the third output the fcsdatscaled array contains the scaled
% parameters. It might be useful for logscaled parameters, but no effect
% in the case of linear parameters. The log scaling is the following
% operation for the "ith" parameter:
% fcsdatscaled(:,i) = ...
%   10.^(fcsdat(:,i)/fcshdr.par(i).range*fcshdr.par(i).decade;);
%
%
% Copyright (c) 2011, Laszlo Balkay
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
%
%     * Redistributions of source code must retain the above copyright
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright
%       notice, this list of conditions and the following disclaimer in
%       the documentation and/or other materials provided with the distribution
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.
%
% Ver 2.5
% 2006-2009 / University of Debrecen, Institute of Nuclear Medicine
% Laszlo Balkay
% balkay@pet.dote.hu
%
% 14/08/2006 I made some changes in the code by the suggestion of
% Brian Harms <brianharms@hotmail.com> and Ivan Cao-Berg <icaoberg@cmu.edu>
% (given at the user reviews area of Mathwork File exchage) The program should work
% in the case of Becton EPics DLM FCS2.0, CyAn Summit FCS3.0 and FACSDiva type
% list mode files.
%
% 29/01/2008 Updated to read the BD LSR II file format and including the comments of
% Allan Moser (Cira Discovery Sciences, Inc.)
%
% 24/01/2009 Updated to read the Partec CyFlow format file. Thanks for
% Gavin A Price
%
% if noarg was supplied

if nargin == 0
    [FileName, FilePath] = uigetfile('*.*','Select fcs file');
    filename = [FilePath,FileName];
    if FileName == 0;
        fcsdat = []; fcshdr = [];
        return;
    end
else
    filecheck = dir(filename);
    if size(filecheck,1) == 0
        hm = msgbox([filename,': The file does not exist!'], ...
            'FcAnalysis info','warn');
        fcsdat = []; fcshdr = [];
        return;
    end
end

% if filename arg. only contain PATH, set the default dir to this
% before issuing the uigetfile command. This is an option for the "fca"
% tool
[FilePath, FileNameMain, fext] = fileparts(filename);
FilePath = [FilePath filesep];
FileName = [FileNameMain, fext];
if  isempty(FileNameMain)
    currend_dir = cd;
    cd(FilePath);
    [FileName, FilePath] = uigetfile('*.*','Select FCS file');
    filename = [FilePath,FileName];
    if FileName == 0;
        fcsdat = []; fcshdr = [];
        return;
    end
    cd(currend_dir);
end

%fid = fopen(filename,'r','ieee-be');
fid = fopen(filename,'r','b');
fcsheader_1stline   = fread(fid,64,'char');
fcsheader_type = char(fcsheader_1stline(1:6)');
%TMP: update to include FCS 3.1
if strcmp(fcsheader_type,'FCS3.1')
    fcsheader_type='FCS3.0';
end
%
%reading the header
%
if strcmp(fcsheader_type,'FCS1.0')
    hm = msgbox('FCS 1.0 file type is not supported!','FcAnalysis info','warn');
    fcsdat = []; fcshdr = [];
    fclose(fid);
    return;
elseif  strcmp(fcsheader_type,'FCS2.0') || strcmp(fcsheader_type,'FCS3.0') % FCS2.0 or FCS3.0 types
    fcshdr.fcstype = fcsheader_type;
    FcsHeaderStartPos   = str2num(char(fcsheader_1stline(11:18)'));
    FcsHeaderStopPos    = str2num(char(fcsheader_1stline(19:26)')); %RLF edited to full 8-byte length
    FcsDataStartPos     = str2num(char(fcsheader_1stline(27:34)')); %RLF edited to full 8-byte length
    status = fseek(fid,FcsHeaderStartPos,'bof');
    fcsheader_main = fread(fid,FcsHeaderStopPos-FcsHeaderStartPos+1,'char');%read the main header
    warning off MATLAB:nonIntegerTruncatedInConversionToChar;
    fcshdr.filename = FileName;
    fcshdr.filepath = FilePath;
    % "The first character of the primary TEXT segment contains the
    % delimiter" (FCS standard)
    if fcsheader_main(1) == 12
        mnemonic_separator = 'FF';
    elseif fcsheader_main(1) == 9
        mnemonic_separator = 'TAB'; %RLF
    else
        mnemonic_separator = char(fcsheader_main(1));
    end
    if mnemonic_separator == '@';% WinMDI
        hm = msgbox([FileName,': The file can not be read (Unsupported FCS type: WinMDI histogram file)'],'FcAnalysis info','warn');
        fcsdat = []; fcshdr = [];
        fclose(fid);
        return;
    end
    fcshdr.TotalEvents = str2num(get_mnemonic_value('$TOT',fcsheader_main, mnemonic_separator));
    if fcshdr.TotalEvents == 0
        fcsdat = 0;
        fcsdatscaled = 0;
        return
    end
    fcshdr.NumOfPar = str2num(get_mnemonic_value('$PAR',fcsheader_main, mnemonic_separator));
    fcshdr.Creator = get_mnemonic_value('CREATOR',fcsheader_main, mnemonic_separator);
    %comp matrix reader added by RLF 12_15_10
    comp = get_mnemonic_value('SPILLOVER',fcsheader_main,mnemonic_separator);
    if ~isempty(comp)
        %%%
        compcell=regexp(comp,',','split');
        nc=str2double(compcell{1});
        fcshdr.CompLabels=compcell(2:nc+1);
        fcshdr.CompMat=reshape(str2double(compcell(nc+2:end)'),[nc nc])';
    else
        fcshdr.CompLabels=[];
        fcshdr.CompMat=[];
    end
    plate = get_mnemonic_value('PLATE NAME',fcsheader_main,mnemonic_separator);
    if ~isempty(plate)
        fcshdr.plate=plate;
    end
    %%%%%%%%%%%%
    
    %%%%%%added by RLF to account for large files
    if FcsDataStartPos == 0
        FcsDataStartPos = str2num(get_mnemonic_value('$BEGINDATA',fcsheader_main, mnemonic_separator));
    end
    %%%%%%%%%%%%%%%%%%%%%
    
    for i=1:fcshdr.NumOfPar
        fcshdr.par(i).name = get_mnemonic_value(['$P',num2str(i),'N'],fcsheader_main, mnemonic_separator);
        fcshdr.par(i).name2 = get_mnemonic_value(['$P',num2str(i),'S'],fcsheader_main, mnemonic_separator);
        fcshdr.par(i).range = str2num(get_mnemonic_value(['$P',num2str(i),'R'],fcsheader_main, mnemonic_separator));
        fcshdr.par(i).bit = str2num(get_mnemonic_value(['$P',num2str(i),'B'],fcsheader_main, mnemonic_separator));
        %==============   Changed way that amplification type is treated ---  ARM  ==================
        par_exponent_str= (get_mnemonic_value(['$P',num2str(i),'E'],fcsheader_main, mnemonic_separator));
        if isempty(par_exponent_str)
            % There is no "$PiE" mnemonic in the Lysys format
            % in that case the PiDISPLAY mnem. shows the LOG or LIN definition
            islogpar = get_mnemonic_value(['P',num2str(i),'DISPLAY'],fcsheader_main, mnemonic_separator);
            if length(islogpar)==3 && isequal(islogpar, 'LOG')  % islogpar == 'LOG'
                par_exponent_str = '5,1';
            else % islogpar = LIN case
                par_exponent_str = '0,0';
            end
        end
        
        par_exponent= str2num(par_exponent_str);
        fcshdr.par(i).decade = par_exponent(1);
        if fcshdr.par(i).decade == 0
            fcshdr.par(i).log = 0;
            fcshdr.par(i).logzero = 0;
        else
            fcshdr.par(i).log = 1;
            if (par_exponent(2) == 0)
                fcshdr.par(i).logzero = 1;
            else
                fcshdr.par(i).logzero = par_exponent(2);
            end
        end
        gain_str = get_mnemonic_value(['$P',num2str(i),'G'],fcsheader_main, mnemonic_separator);
        if ~isempty(gain_str)
            fcshdr.par(i).gain=str2double(gain_str);
        else
            fcshdr.par(i).gain=1;
        end
        
        %============================================================================================
    end
    fcshdr.starttime = get_mnemonic_value('$BTIM',fcsheader_main, mnemonic_separator);
    fcshdr.stoptime = get_mnemonic_value('$ETIM',fcsheader_main, mnemonic_separator);
    fcshdr.cytometry = get_mnemonic_value('$CYT',fcsheader_main, mnemonic_separator);
    fcshdr.date = get_mnemonic_value('$DATE',fcsheader_main, mnemonic_separator);
    fcshdr.byteorder = get_mnemonic_value('$BYTEORD',fcsheader_main, mnemonic_separator);
    fcshdr.datatype = get_mnemonic_value('$DATATYPE',fcsheader_main, mnemonic_separator);
    fcshdr.system = get_mnemonic_value('$SYS',fcsheader_main, mnemonic_separator);
    fcshdr.project = get_mnemonic_value('$PROJ',fcsheader_main, mnemonic_separator);
    fcshdr.experiment = get_mnemonic_value('$EXP',fcsheader_main, mnemonic_separator);
    fcshdr.cells = get_mnemonic_value('$Cells',fcsheader_main, mnemonic_separator);
    fcshdr.creator = get_mnemonic_value('CREATOR',fcsheader_main, mnemonic_separator);
    fcshdr.cytsn = get_mnemonic_value('$CYTSN',fcsheader_main, mnemonic_separator);
else
    hm = msgbox([FileName,': The file can not be read (Unsupported FCS type)'],'FcAnalysis info','warn');
    fcsdat = []; fcshdr = [];
    fclose(fid);
    return;
end
%
%reading the events
%
status = fseek(fid,FcsDataStartPos,'bof');
if strcmp(fcsheader_type,'FCS2.0')
    if strcmp(mnemonic_separator,'\') || strcmp(mnemonic_separator,'FF')... %ordinary or FacsDIVA FCS2.0
            || strcmp(mnemonic_separator,'/') || strcmp(mnemonic_separator,'TAB')% added by GAP 1/22/09 %added by RLF 09/02/10
        if fcshdr.par(1).bit == 16
            fcsdat = uint16(fread(fid,[fcshdr.NumOfPar fcshdr.TotalEvents],'uint16')');
            if strcmp(fcshdr.byteorder,'1,2')...% this is the Cytomics data
                    || strcmp(fcshdr.byteorder, '1,2,3,4') %added by GAP 1/22/09
                fcsdat = bitor(bitshift(fcsdat,-8),bitshift(fcsdat,8));
            end
        elseif fcshdr.par(1).bit == 32
            if fcshdr.datatype ~= 'F'
                fcsdat = (fread(fid,[fcshdr.NumOfPar fcshdr.TotalEvents],'uint32')');
            else % 'LYSYS' case
                fcsdat = (fread(fid,[fcshdr.NumOfPar fcshdr.TotalEvents],'float32')');
            end
        else
            bittype = ['ubit',num2str(fcshdr.par(1).bit)];
            fcsdat = fread(fid,[fcshdr.NumOfPar fcshdr.TotalEvents],bittype, 'ieee-le')';
        end
    elseif strcmp(mnemonic_separator,'!');% Becton EPics DLM FCS2.0
        fcsdat_ = fread(fid,[fcshdr.NumOfPar fcshdr.TotalEvents],'uint16', 'ieee-le')';
        fcsdat = zeros(fcshdr.TotalEvents,fcshdr.NumOfPar);
        for i=1:fcshdr.NumOfPar
            bintmp = dec2bin(fcsdat_(:,i));
            fcsdat(:,i) = bin2dec(bintmp(:,7:16)); % only the first 10bit is valid for the parameter
        end
    end
    fclose(fid);
elseif strcmp(fcsheader_type,'FCS3.0')
    %     if strcmp(mnemonic_separator,'|') % CyAn Summit FCS3.0
    %         fcsdat_ = (fread(fid,[fcshdr.NumOfPar fcshdr.TotalEvents],'uint16','ieee-le')');
    %         fcsdat = zeros(size(fcsdat_));
    %         new_xrange = 1024;
    %         for i=1:fcshdr.NumOfPar
    %             fcsdat(:,i) = fcsdat_(:,i)*new_xrange/fcshdr.par(i).range;
    %             fcshdr.par(i).range = new_xrange;
    %         end
    %     else % ordinary FCS 3.0
    %%%%%edited by RLF 06_30_10
    if strcmp(fcshdr.datatype,'D')
        if strcmp(fcshdr.byteorder, '1,2,3,4')
            fcsdat = fread(fid,[fcshdr.NumOfPar fcshdr.TotalEvents],'double','l')';
        elseif strcmp(fcshdr.byteorder,'4,3,2,1')
            fcsdat = fread(fid,[fcshdr.NumOfPar fcshdr.TotalEvents],'double','b')';
        end
    elseif strcmp(fcshdr.datatype,'F')
        if strcmp(fcshdr.byteorder, '1,2,3,4')
            fcsdat = fread(fid,[fcshdr.NumOfPar fcshdr.TotalEvents],'float32','l')';
        elseif strcmp(fcshdr.byteorder,'4,3,2,1')
            fcsdat = fread(fid,[fcshdr.NumOfPar fcshdr.TotalEvents],'float32','b')';
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%
    %     end
    fclose(fid);
end
%
%calculate the scaled events (for log scales) %RLF added gain division
if nargout>2
    fcsdatscaled = zeros(size(fcsdat));
    for  i = 1 : fcshdr.NumOfPar
        Xlogdecade = fcshdr.par(i).decade;
        XChannelMax = fcshdr.par(i).range;
        Xlogvalatzero = fcshdr.par(i).logzero;
        if fcshdr.par(i).gain~=1
            fcsdatscaled(:,i)  = double(fcsdat(:,i))./fcshdr.par(i).gain;
            
        elseif fcshdr.par(i).log
            fcsdatscaled(:,i) = Xlogvalatzero*10.^(double(fcsdat(:,i))/XChannelMax*Xlogdecade);
        else fcsdatscaled(:,i)  = fcsdat(:,i);
        end
    end
    
end

if nargout>3 && ~isempty(fcshdr.CompLabels) %RLF. applied to fcsdatscaled rather than fcsdat.
    
    compcols=zeros(1,nc);
    colLabels={fcshdr.par.name};
    for i=1:nc
        compcols(i)=find(strcmp(fcshdr.CompLabels{i},colLabels));
    end
    fcsdatcomp=fcsdatscaled;
    fcsdatcomp(:,compcols)=fcsdatcomp(:,compcols)/fcshdr.CompMat;
else fcsdatcomp=[];
end
end
%%%%
function mneval = get_mnemonic_value(mnemonic_name,fcsheader,mnemonic_separator)

if strcmp(mnemonic_separator,'\')  || strcmp(mnemonic_separator,'!') ...
        || strcmp(mnemonic_separator,'|') || strcmp(mnemonic_separator,'@')...
        || strcmp(mnemonic_separator, '/') % added by GAP 1/22/08
    mnemonic_startpos = findstr(char(fcsheader'),mnemonic_name);
    if isempty(mnemonic_startpos)
        mneval = [];
        return;
    end
    mnemonic_length = length(mnemonic_name);
    mnemonic_stoppos = mnemonic_startpos + mnemonic_length;
    next_slashes = findstr(char(fcsheader(mnemonic_stoppos+1:end)'),mnemonic_separator);
    next_slash = next_slashes(1) + mnemonic_stoppos;
    
    mneval = char(fcsheader(mnemonic_stoppos+1:next_slash-1)');
elseif strcmp(mnemonic_separator,'FF')
    mnemonic_startpos = findstr(char(fcsheader'),mnemonic_name);
    if isempty(mnemonic_startpos)
        mneval = [];
        return;
    end
    mnemonic_length = length(mnemonic_name);
    mnemonic_stoppos = mnemonic_startpos + mnemonic_length ;
    next_formfeeds = find( fcsheader(mnemonic_stoppos+1:end) == 12);
    next_formfeed = next_formfeeds(1) + mnemonic_stoppos;
    
    mneval = char(fcsheader(mnemonic_stoppos + 1 : next_formfeed-1)');
elseif strcmp(mnemonic_separator,'TAB') %added by RLF August 2010
    mnemonic_startpos = findstr(char(fcsheader'),mnemonic_name);
    if isempty(mnemonic_startpos)
        mneval = [];
        return;
    end
    mnemonic_length = length(mnemonic_name);
    mnemonic_stoppos = mnemonic_startpos + mnemonic_length ;
    next_formfeeds = find( fcsheader(mnemonic_stoppos+1:end) == 9);
    next_formfeed = next_formfeeds(1) + mnemonic_stoppos;
    
    mneval = char(fcsheader(mnemonic_stoppos + 1 : next_formfeed-1)');
end
end
%%
function [cs,index] = sort_nat(c,mode)
%sort_nat: Natural order sort of cell array of strings.
% usage:  [S,INDEX] = sort_nat(C)
%
% where,
%    C is a cell array (vector) of strings to be sorted.
%    S is C, sorted in natural order.
%    INDEX is the sort order such that S = C(INDEX);
%
% Natural order sorting sorts strings containing digits in a way such that
% the numerical value of the digits is taken into account.  It is
% especially useful for sorting file names containing index numbers with
% different numbers of digits.  Often, people will use leading zeros to get
% the right sort order, but with this function you don't have to do that.
% For example, if C = {'file1.txt','file2.txt','file10.txt'}, a normal sort
% will give you
%
%       {'file1.txt'  'file10.txt'  'file2.txt'}
%
% whereas, sort_nat will give you
%
%       {'file1.txt'  'file2.txt'  'file10.txt'}
%
% See also: sort

% Version: 1.4, 22 January 2011
% Author:  Douglas M. Schwarz
% Email:   dmschwarz=ieee*org, dmschwarz=urgrad*rochester*edu
% Real_email = regexprep(Email,{'=','*'},{'@','.'})


% Set default value for mode if necessary.
if nargin < 2
    mode = 'ascend';
end

% Make sure mode is either 'ascend' or 'descend'.
modes = strcmpi(mode,{'ascend','descend'});
is_descend = modes(2);
if ~any(modes)
    error('sort_nat:sortDirection',...
        'sorting direction must be ''ascend'' or ''descend''.')
end

% Replace runs of digits with '0'.
c2 = regexprep(c,'\d+','0');

% Compute char version of c2 and locations of zeros.
s1 = char(c2);
z = s1 == '0';

% Extract the runs of digits and their start and end indices.
[digruns,first,last] = regexp(c,'\d+','match','start','end');

% Create matrix of numerical values of runs of digits and a matrix of the
% number of digits in each run.
num_str = length(c);
max_len = size(s1,2);
num_val = NaN(num_str,max_len);
num_dig = NaN(num_str,max_len);
for i = 1:num_str
    num_val(i,z(i,:)) = sscanf(sprintf('%s ',digruns{i}{:}),'%f');
    num_dig(i,z(i,:)) = last{i} - first{i} + 1;
end

% Find columns that have at least one non-NaN.  Make sure activecols is a
% 1-by-n vector even if n = 0.
activecols = reshape(find(~all(isnan(num_val))),1,[]);
n = length(activecols);

% Compute which columns in the composite matrix get the numbers.
numcols = activecols + (1:2:2*n);

% Compute which columns in the composite matrix get the number of digits.
ndigcols = numcols + 1;

% Compute which columns in the composite matrix get chars.
charcols = true(1,max_len + 2*n);
charcols(numcols) = false;
charcols(ndigcols) = false;

% Create and fill composite matrix, comp.
comp = zeros(num_str,max_len + 2*n);
comp(:,charcols) = double(s1);
comp(:,numcols) = num_val(:,activecols);
comp(:,ndigcols) = num_dig(:,activecols);

% Sort rows of composite matrix and use index to sort c in ascending or
% descending order, depending on mode.
[unused,index] = sortrows(comp);
if is_descend
    index = index(end:-1:1);
end
index = reshape(index,size(c));
cs = c(index);
end