??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
-
Sqrt
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.12v2.7.0-217-g2a0f59ecfe68??
\
meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namemean
U
mean/Read/ReadVariableOpReadVariableOpmean*
_output_shapes
: *
dtype0
d
varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
variance
]
variance/Read/ReadVariableOpReadVariableOpvariance*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0	
`
mean_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namemean_1
Y
mean_1/Read/ReadVariableOpReadVariableOpmean_1*
_output_shapes
: *
dtype0
h

variance_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
variance_1
a
variance_1/Read/ReadVariableOpReadVariableOp
variance_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0	
`
mean_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namemean_2
Y
mean_2/Read/ReadVariableOpReadVariableOpmean_2*
_output_shapes
: *
dtype0
h

variance_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
variance_2
a
variance_2/Read/ReadVariableOpReadVariableOp
variance_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0	
`
mean_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namemean_3
Y
mean_3/Read/ReadVariableOpReadVariableOpmean_3*
_output_shapes
: *
dtype0
h

variance_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
variance_3
a
variance_3/Read/ReadVariableOpReadVariableOp
variance_3*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0	
`
mean_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namemean_4
Y
mean_4/Read/ReadVariableOpReadVariableOpmean_4*
_output_shapes
: *
dtype0
h

variance_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
variance_4
a
variance_4/Read/ReadVariableOpReadVariableOp
variance_4*
_output_shapes
: *
dtype0
b
count_4VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	count_4
[
count_4/Read/ReadVariableOpReadVariableOpcount_4*
_output_shapes
: *
dtype0	
`
mean_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namemean_5
Y
mean_5/Read/ReadVariableOpReadVariableOpmean_5*
_output_shapes
: *
dtype0
h

variance_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
variance_5
a
variance_5/Read/ReadVariableOpReadVariableOp
variance_5*
_output_shapes
: *
dtype0
b
count_5VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	count_5
[
count_5/Read/ReadVariableOpReadVariableOpcount_5*
_output_shapes
: *
dtype0	
`
mean_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namemean_6
Y
mean_6/Read/ReadVariableOpReadVariableOpmean_6*
_output_shapes
: *
dtype0
h

variance_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
variance_6
a
variance_6/Read/ReadVariableOpReadVariableOp
variance_6*
_output_shapes
: *
dtype0
b
count_6VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	count_6
[
count_6/Read/ReadVariableOpReadVariableOpcount_6*
_output_shapes
: *
dtype0	
`
mean_7VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namemean_7
Y
mean_7/Read/ReadVariableOpReadVariableOpmean_7*
_output_shapes
: *
dtype0
h

variance_7VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
variance_7
a
variance_7/Read/ReadVariableOpReadVariableOp
variance_7*
_output_shapes
: *
dtype0
b
count_7VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	count_7
[
count_7/Read/ReadVariableOpReadVariableOpcount_7*
_output_shapes
: *
dtype0	
`
mean_8VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namemean_8
Y
mean_8/Read/ReadVariableOpReadVariableOpmean_8*
_output_shapes
: *
dtype0
h

variance_8VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
variance_8
a
variance_8/Read/ReadVariableOpReadVariableOp
variance_8*
_output_shapes
: *
dtype0
b
count_8VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	count_8
[
count_8/Read/ReadVariableOpReadVariableOpcount_8*
_output_shapes
: *
dtype0	
`
mean_9VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namemean_9
Y
mean_9/Read/ReadVariableOpReadVariableOpmean_9*
_output_shapes
: *
dtype0
h

variance_9VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
variance_9
a
variance_9/Read/ReadVariableOpReadVariableOp
variance_9*
_output_shapes
: *
dtype0
b
count_9VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	count_9
[
count_9/Read/ReadVariableOpReadVariableOpcount_9*
_output_shapes
: *
dtype0	
b
mean_10VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	mean_10
[
mean_10/Read/ReadVariableOpReadVariableOpmean_10*
_output_shapes
: *
dtype0
j
variance_10VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namevariance_10
c
variance_10/Read/ReadVariableOpReadVariableOpvariance_10*
_output_shapes
: *
dtype0
d
count_10VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
count_10
]
count_10/Read/ReadVariableOpReadVariableOpcount_10*
_output_shapes
: *
dtype0	
b
mean_11VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	mean_11
[
mean_11/Read/ReadVariableOpReadVariableOpmean_11*
_output_shapes
: *
dtype0
j
variance_11VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namevariance_11
c
variance_11/Read/ReadVariableOpReadVariableOpvariance_11*
_output_shapes
: *
dtype0
d
count_11VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
count_11
]
count_11/Read/ReadVariableOpReadVariableOpcount_11*
_output_shapes
: *
dtype0	
b
mean_12VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	mean_12
[
mean_12/Read/ReadVariableOpReadVariableOpmean_12*
_output_shapes
: *
dtype0
j
variance_12VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namevariance_12
c
variance_12/Read/ReadVariableOpReadVariableOpvariance_12*
_output_shapes
: *
dtype0
d
count_12VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
count_12
]
count_12/Read/ReadVariableOpReadVariableOpcount_12*
_output_shapes
: *
dtype0	
b
mean_13VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	mean_13
[
mean_13/Read/ReadVariableOpReadVariableOpmean_13*
_output_shapes
: *
dtype0
j
variance_13VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namevariance_13
c
variance_13/Read/ReadVariableOpReadVariableOpvariance_13*
_output_shapes
: *
dtype0
d
count_13VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
count_13
]
count_13/Read/ReadVariableOpReadVariableOpcount_13*
_output_shapes
: *
dtype0	
b
mean_14VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	mean_14
[
mean_14/Read/ReadVariableOpReadVariableOpmean_14*
_output_shapes
: *
dtype0
j
variance_14VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namevariance_14
c
variance_14/Read/ReadVariableOpReadVariableOpvariance_14*
_output_shapes
: *
dtype0
d
count_14VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
count_14
]
count_14/Read/ReadVariableOpReadVariableOpcount_14*
_output_shapes
: *
dtype0	
b
mean_15VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	mean_15
[
mean_15/Read/ReadVariableOpReadVariableOpmean_15*
_output_shapes
: *
dtype0
j
variance_15VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namevariance_15
c
variance_15/Read/ReadVariableOpReadVariableOpvariance_15*
_output_shapes
: *
dtype0
d
count_15VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
count_15
]
count_15/Read/ReadVariableOpReadVariableOpcount_15*
_output_shapes
: *
dtype0	
b
mean_16VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	mean_16
[
mean_16/Read/ReadVariableOpReadVariableOpmean_16*
_output_shapes
: *
dtype0
j
variance_16VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namevariance_16
c
variance_16/Read/ReadVariableOpReadVariableOpvariance_16*
_output_shapes
: *
dtype0
d
count_16VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
count_16
]
count_16/Read/ReadVariableOpReadVariableOpcount_16*
_output_shapes
: *
dtype0	
b
mean_17VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	mean_17
[
mean_17/Read/ReadVariableOpReadVariableOpmean_17*
_output_shapes
: *
dtype0
j
variance_17VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namevariance_17
c
variance_17/Read/ReadVariableOpReadVariableOpvariance_17*
_output_shapes
: *
dtype0
d
count_17VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
count_17
]
count_17/Read/ReadVariableOpReadVariableOpcount_17*
_output_shapes
: *
dtype0	
b
mean_18VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	mean_18
[
mean_18/Read/ReadVariableOpReadVariableOpmean_18*
_output_shapes
: *
dtype0
j
variance_18VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namevariance_18
c
variance_18/Read/ReadVariableOpReadVariableOpvariance_18*
_output_shapes
: *
dtype0
d
count_18VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
count_18
]
count_18/Read/ReadVariableOpReadVariableOpcount_18*
_output_shapes
: *
dtype0	
b
mean_19VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	mean_19
[
mean_19/Read/ReadVariableOpReadVariableOpmean_19*
_output_shapes
: *
dtype0
j
variance_19VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namevariance_19
c
variance_19/Read/ReadVariableOpReadVariableOpvariance_19*
_output_shapes
: *
dtype0
d
count_19VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
count_19
]
count_19/Read/ReadVariableOpReadVariableOpcount_19*
_output_shapes
: *
dtype0	
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

: *
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
: *
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

: *
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
d
count_20VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_20
]
count_20/Read/ReadVariableOpReadVariableOpcount_20*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
d
count_21VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_21
]
count_21/Read/ReadVariableOpReadVariableOpcount_21*
_output_shapes
: *
dtype0
R
ConstConst*
_output_shapes
:*
dtype0*
valueB*G?D
T
Const_1Const*
_output_shapes
:*
dtype0*
valueB*??;H
T
Const_2Const*
_output_shapes
:*
dtype0*
valueB*?( ?
T
Const_3Const*
_output_shapes
:*
dtype0*
valueB*??>
T
Const_4Const*
_output_shapes
:*
dtype0*
valueB*2???
T
Const_5Const*
_output_shapes
:*
dtype0*
valueB*t?)?
T
Const_6Const*
_output_shapes
:*
dtype0*
valueB*[??
T
Const_7Const*
_output_shapes
:*
dtype0*
valueB*??>
T
Const_8Const*
_output_shapes
:*
dtype0*
valueB*Q??@
T
Const_9Const*
_output_shapes
:*
dtype0*
valueB*?W?A
U
Const_10Const*
_output_shapes
:*
dtype0*
valueB*?G?
U
Const_11Const*
_output_shapes
:*
dtype0*
valueB*}?>
U
Const_12Const*
_output_shapes
:*
dtype0*
valueB*N?A
U
Const_13Const*
_output_shapes
:*
dtype0*
valueB*?ХC
U
Const_14Const*
_output_shapes
:*
dtype0*
valueB*2 ?
U
Const_15Const*
_output_shapes
:*
dtype0*
valueB*C??=
U
Const_16Const*
_output_shapes
:*
dtype0*
valueB*?xC
U
Const_17Const*
_output_shapes
:*
dtype0*
valueB*??D
U
Const_18Const*
_output_shapes
:*
dtype0*
valueB*?G?@
U
Const_19Const*
_output_shapes
:*
dtype0*
valueB*???@
U
Const_20Const*
_output_shapes
:*
dtype0*
valueB*գA
U
Const_21Const*
_output_shapes
:*
dtype0*
valueB*??B
U
Const_22Const*
_output_shapes
:*
dtype0*
valueB*?'D
U
Const_23Const*
_output_shapes
:*
dtype0*
valueB*B(?H
U
Const_24Const*
_output_shapes
:*
dtype0*
valueB*\?D
U
Const_25Const*
_output_shapes
:*
dtype0*
valueB*?]:H
U
Const_26Const*
_output_shapes
:*
dtype0*
valueB*??E
U
Const_27Const*
_output_shapes
:*
dtype0*
valueB*Qz?I
U
Const_28Const*
_output_shapes
:*
dtype0*
valueB*??DA
U
Const_29Const*
_output_shapes
:*
dtype0*
valueB*̻?A
U
Const_30Const*
_output_shapes
:*
dtype0*
valueB*?Q?@
U
Const_31Const*
_output_shapes
:*
dtype0*
valueB*d??A
U
Const_32Const*
_output_shapes
:*
dtype0*
valueB*??1A
U
Const_33Const*
_output_shapes
:*
dtype0*
valueB*???A
U
Const_34Const*
_output_shapes
:*
dtype0*
valueB*43C?
U
Const_35Const*
_output_shapes
:*
dtype0*
valueB*?p9>
U
Const_36Const*
_output_shapes
:*
dtype0*
valueB*\??
U
Const_37Const*
_output_shapes
:*
dtype0*
valueB*??>
U
Const_38Const*
_output_shapes
:*
dtype0*
valueB*?Q ?
U
Const_39Const*
_output_shapes
:*
dtype0*
valueB*??>

NoOpNoOp
?m
Const_40Const"/device:CPU:0*
_output_shapes
: *
dtype0*?m
value?mB?m B?m
?

layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer_with_weights-0
layer-20
layer_with_weights-1
layer-21
layer_with_weights-2
layer-22
layer_with_weights-3
layer-23
layer_with_weights-4
layer-24
layer_with_weights-5
layer-25
layer_with_weights-6
layer-26
layer_with_weights-7
layer-27
layer_with_weights-8
layer-28
layer_with_weights-9
layer-29
layer_with_weights-10
layer-30
 layer_with_weights-11
 layer-31
!layer_with_weights-12
!layer-32
"layer_with_weights-13
"layer-33
#layer_with_weights-14
#layer-34
$layer_with_weights-15
$layer-35
%layer_with_weights-16
%layer-36
&layer_with_weights-17
&layer-37
'layer_with_weights-18
'layer-38
(layer_with_weights-19
(layer-39
)layer-40
*layer_with_weights-20
*layer-41
+layer-42
,layer_with_weights-21
,layer-43
-	optimizer
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2
signatures
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
3
_keep_axis
4_reduce_axis
5_reduce_axis_mask
6_broadcast_shape
7mean
7
adapt_mean
8variance
8adapt_variance
	9count
:	keras_api
?
;
_keep_axis
<_reduce_axis
=_reduce_axis_mask
>_broadcast_shape
?mean
?
adapt_mean
@variance
@adapt_variance
	Acount
B	keras_api
?
C
_keep_axis
D_reduce_axis
E_reduce_axis_mask
F_broadcast_shape
Gmean
G
adapt_mean
Hvariance
Hadapt_variance
	Icount
J	keras_api
?
K
_keep_axis
L_reduce_axis
M_reduce_axis_mask
N_broadcast_shape
Omean
O
adapt_mean
Pvariance
Padapt_variance
	Qcount
R	keras_api
?
S
_keep_axis
T_reduce_axis
U_reduce_axis_mask
V_broadcast_shape
Wmean
W
adapt_mean
Xvariance
Xadapt_variance
	Ycount
Z	keras_api
?
[
_keep_axis
\_reduce_axis
]_reduce_axis_mask
^_broadcast_shape
_mean
_
adapt_mean
`variance
`adapt_variance
	acount
b	keras_api
?
c
_keep_axis
d_reduce_axis
e_reduce_axis_mask
f_broadcast_shape
gmean
g
adapt_mean
hvariance
hadapt_variance
	icount
j	keras_api
?
k
_keep_axis
l_reduce_axis
m_reduce_axis_mask
n_broadcast_shape
omean
o
adapt_mean
pvariance
padapt_variance
	qcount
r	keras_api
?
s
_keep_axis
t_reduce_axis
u_reduce_axis_mask
v_broadcast_shape
wmean
w
adapt_mean
xvariance
xadapt_variance
	ycount
z	keras_api
?
{
_keep_axis
|_reduce_axis
}_reduce_axis_mask
~_broadcast_shape
mean

adapt_mean
?variance
?adapt_variance

?count
?	keras_api
?
?
_keep_axis
?_reduce_axis
?_reduce_axis_mask
?_broadcast_shape
	?mean
?
adapt_mean
?variance
?adapt_variance

?count
?	keras_api
?
?
_keep_axis
?_reduce_axis
?_reduce_axis_mask
?_broadcast_shape
	?mean
?
adapt_mean
?variance
?adapt_variance

?count
?	keras_api
?
?
_keep_axis
?_reduce_axis
?_reduce_axis_mask
?_broadcast_shape
	?mean
?
adapt_mean
?variance
?adapt_variance

?count
?	keras_api
?
?
_keep_axis
?_reduce_axis
?_reduce_axis_mask
?_broadcast_shape
	?mean
?
adapt_mean
?variance
?adapt_variance

?count
?	keras_api
?
?
_keep_axis
?_reduce_axis
?_reduce_axis_mask
?_broadcast_shape
	?mean
?
adapt_mean
?variance
?adapt_variance

?count
?	keras_api
?
?
_keep_axis
?_reduce_axis
?_reduce_axis_mask
?_broadcast_shape
	?mean
?
adapt_mean
?variance
?adapt_variance

?count
?	keras_api
?
?
_keep_axis
?_reduce_axis
?_reduce_axis_mask
?_broadcast_shape
	?mean
?
adapt_mean
?variance
?adapt_variance

?count
?	keras_api
?
?
_keep_axis
?_reduce_axis
?_reduce_axis_mask
?_broadcast_shape
	?mean
?
adapt_mean
?variance
?adapt_variance

?count
?	keras_api
?
?
_keep_axis
?_reduce_axis
?_reduce_axis_mask
?_broadcast_shape
	?mean
?
adapt_mean
?variance
?adapt_variance

?count
?	keras_api
?
?
_keep_axis
?_reduce_axis
?_reduce_axis_mask
?_broadcast_shape
	?mean
?
adapt_mean
?variance
?adapt_variance

?count
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
 
?
70
81
92
?3
@4
A5
G6
H7
I8
O9
P10
Q11
W12
X13
Y14
_15
`16
a17
g18
h19
i20
o21
p22
q23
w24
x25
y26
27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?50
?51
?52
?53
?54
?55
?56
?57
?58
?59
?60
?61
?62
?63
 
?0
?1
?2
?3
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
.	variables
/trainable_variables
0regularization_losses
 
 
 
 
 
NL
VARIABLE_VALUEmean4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEvariance8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEcount5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
PN
VARIABLE_VALUEmean_14layer_with_weights-1/mean/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUE
variance_18layer_with_weights-1/variance/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_15layer_with_weights-1/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
PN
VARIABLE_VALUEmean_24layer_with_weights-2/mean/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUE
variance_28layer_with_weights-2/variance/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_25layer_with_weights-2/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
PN
VARIABLE_VALUEmean_34layer_with_weights-3/mean/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUE
variance_38layer_with_weights-3/variance/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_35layer_with_weights-3/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
PN
VARIABLE_VALUEmean_44layer_with_weights-4/mean/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUE
variance_48layer_with_weights-4/variance/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_45layer_with_weights-4/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
PN
VARIABLE_VALUEmean_54layer_with_weights-5/mean/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUE
variance_58layer_with_weights-5/variance/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_55layer_with_weights-5/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
PN
VARIABLE_VALUEmean_64layer_with_weights-6/mean/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUE
variance_68layer_with_weights-6/variance/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_65layer_with_weights-6/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
PN
VARIABLE_VALUEmean_74layer_with_weights-7/mean/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUE
variance_78layer_with_weights-7/variance/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_75layer_with_weights-7/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
PN
VARIABLE_VALUEmean_84layer_with_weights-8/mean/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUE
variance_88layer_with_weights-8/variance/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_85layer_with_weights-8/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
PN
VARIABLE_VALUEmean_94layer_with_weights-9/mean/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUE
variance_98layer_with_weights-9/variance/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_95layer_with_weights-9/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
RP
VARIABLE_VALUEmean_105layer_with_weights-10/mean/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEvariance_109layer_with_weights-10/variance/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEcount_106layer_with_weights-10/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
RP
VARIABLE_VALUEmean_115layer_with_weights-11/mean/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEvariance_119layer_with_weights-11/variance/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEcount_116layer_with_weights-11/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
RP
VARIABLE_VALUEmean_125layer_with_weights-12/mean/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEvariance_129layer_with_weights-12/variance/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEcount_126layer_with_weights-12/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
RP
VARIABLE_VALUEmean_135layer_with_weights-13/mean/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEvariance_139layer_with_weights-13/variance/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEcount_136layer_with_weights-13/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
RP
VARIABLE_VALUEmean_145layer_with_weights-14/mean/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEvariance_149layer_with_weights-14/variance/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEcount_146layer_with_weights-14/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
RP
VARIABLE_VALUEmean_155layer_with_weights-15/mean/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEvariance_159layer_with_weights-15/variance/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEcount_156layer_with_weights-15/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
RP
VARIABLE_VALUEmean_165layer_with_weights-16/mean/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEvariance_169layer_with_weights-16/variance/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEcount_166layer_with_weights-16/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
RP
VARIABLE_VALUEmean_175layer_with_weights-17/mean/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEvariance_179layer_with_weights-17/variance/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEcount_176layer_with_weights-17/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
RP
VARIABLE_VALUEmean_185layer_with_weights-18/mean/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEvariance_189layer_with_weights-18/variance/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEcount_186layer_with_weights-18/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
RP
VARIABLE_VALUEmean_195layer_with_weights-19/mean/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEvariance_199layer_with_weights-19/variance/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEcount_196layer_with_weights-19/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
YW
VARIABLE_VALUEdense/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
dense/bias5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
[Y
VARIABLE_VALUEdense_1/kernel7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_1/bias5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?
70
81
92
?3
@4
A5
G6
H7
I8
O9
P10
Q11
W12
X13
Y14
_15
`16
a17
g18
h19
i20
o21
p22
q23
w24
x25
y26
27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?50
?51
?52
?53
?54
?55
?56
?57
?58
?59
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_204keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEcount_214keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
?
serving_default_battery_powerPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
w
serving_default_bluePlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
~
serving_default_clock_speedPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_dual_simPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
u
serving_default_fcPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
y
serving_default_four_gPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
}
serving_default_int_memoryPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
x
serving_default_m_depPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
|
serving_default_mobile_wtPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
z
serving_default_n_coresPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
u
serving_default_pcPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
|
serving_default_px_heightPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_px_widthPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
v
serving_default_ramPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
w
serving_default_sc_hPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
w
serving_default_sc_wPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
|
serving_default_talk_timePlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
z
serving_default_three_gPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????

serving_default_touch_screenPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
w
serving_default_wifiPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?

StatefulPartitionedCallStatefulPartitionedCallserving_default_battery_powerserving_default_blueserving_default_clock_speedserving_default_dual_simserving_default_fcserving_default_four_gserving_default_int_memoryserving_default_m_depserving_default_mobile_wtserving_default_n_coresserving_default_pcserving_default_px_heightserving_default_px_widthserving_default_ramserving_default_sc_hserving_default_sc_wserving_default_talk_timeserving_default_three_gserving_default_touch_screenserving_default_wifiConstConst_1Const_2Const_3Const_4Const_5Const_6Const_7Const_8Const_9Const_10Const_11Const_12Const_13Const_14Const_15Const_16Const_17Const_18Const_19Const_20Const_21Const_22Const_23Const_24Const_25Const_26Const_27Const_28Const_29Const_30Const_31Const_32Const_33Const_34Const_35Const_36Const_37Const_38Const_39dense/kernel
dense/biasdense_1/kerneldense_1/bias*K
TinD
B2@*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
<=>?*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_9098
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount/Read/ReadVariableOpmean_1/Read/ReadVariableOpvariance_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpmean_2/Read/ReadVariableOpvariance_2/Read/ReadVariableOpcount_2/Read/ReadVariableOpmean_3/Read/ReadVariableOpvariance_3/Read/ReadVariableOpcount_3/Read/ReadVariableOpmean_4/Read/ReadVariableOpvariance_4/Read/ReadVariableOpcount_4/Read/ReadVariableOpmean_5/Read/ReadVariableOpvariance_5/Read/ReadVariableOpcount_5/Read/ReadVariableOpmean_6/Read/ReadVariableOpvariance_6/Read/ReadVariableOpcount_6/Read/ReadVariableOpmean_7/Read/ReadVariableOpvariance_7/Read/ReadVariableOpcount_7/Read/ReadVariableOpmean_8/Read/ReadVariableOpvariance_8/Read/ReadVariableOpcount_8/Read/ReadVariableOpmean_9/Read/ReadVariableOpvariance_9/Read/ReadVariableOpcount_9/Read/ReadVariableOpmean_10/Read/ReadVariableOpvariance_10/Read/ReadVariableOpcount_10/Read/ReadVariableOpmean_11/Read/ReadVariableOpvariance_11/Read/ReadVariableOpcount_11/Read/ReadVariableOpmean_12/Read/ReadVariableOpvariance_12/Read/ReadVariableOpcount_12/Read/ReadVariableOpmean_13/Read/ReadVariableOpvariance_13/Read/ReadVariableOpcount_13/Read/ReadVariableOpmean_14/Read/ReadVariableOpvariance_14/Read/ReadVariableOpcount_14/Read/ReadVariableOpmean_15/Read/ReadVariableOpvariance_15/Read/ReadVariableOpcount_15/Read/ReadVariableOpmean_16/Read/ReadVariableOpvariance_16/Read/ReadVariableOpcount_16/Read/ReadVariableOpmean_17/Read/ReadVariableOpvariance_17/Read/ReadVariableOpcount_17/Read/ReadVariableOpmean_18/Read/ReadVariableOpvariance_18/Read/ReadVariableOpcount_18/Read/ReadVariableOpmean_19/Read/ReadVariableOpvariance_19/Read/ReadVariableOpcount_19/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount_20/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_21/Read/ReadVariableOpConst_40*Q
TinJ
H2F																				*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_11029
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemeanvariancecountmean_1
variance_1count_1mean_2
variance_2count_2mean_3
variance_3count_3mean_4
variance_4count_4mean_5
variance_5count_5mean_6
variance_6count_6mean_7
variance_7count_7mean_8
variance_8count_8mean_9
variance_9count_9mean_10variance_10count_10mean_11variance_11count_11mean_12variance_12count_12mean_13variance_13count_13mean_14variance_14count_14mean_15variance_15count_15mean_16variance_16count_16mean_17variance_17count_17mean_18variance_18count_18mean_19variance_19count_19dense/kernel
dense/biasdense_1/kerneldense_1/biastotalcount_20total_1count_21*P
TinI
G2E*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_11243??
??
?
?__inference_model_layer_call_and_return_conditional_losses_8809
battery_power
blue
clock_speed
dual_sim
fc

four_g

int_memory	
m_dep
	mobile_wt
n_cores
pc
	px_height
px_width
ram
sc_h
sc_w
	talk_time
three_g
touch_screen
wifi
normalization_sub_y
normalization_sqrt_x
normalization_1_sub_y
normalization_1_sqrt_x
normalization_2_sub_y
normalization_2_sqrt_x
normalization_3_sub_y
normalization_3_sqrt_x
normalization_4_sub_y
normalization_4_sqrt_x
normalization_5_sub_y
normalization_5_sqrt_x
normalization_6_sub_y
normalization_6_sqrt_x
normalization_7_sub_y
normalization_7_sqrt_x
normalization_8_sub_y
normalization_8_sqrt_x
normalization_9_sub_y
normalization_9_sqrt_x
normalization_10_sub_y
normalization_10_sqrt_x
normalization_11_sub_y
normalization_11_sqrt_x
normalization_12_sub_y
normalization_12_sqrt_x
normalization_13_sub_y
normalization_13_sqrt_x
normalization_14_sub_y
normalization_14_sqrt_x
normalization_15_sub_y
normalization_15_sqrt_x
normalization_16_sub_y
normalization_16_sqrt_x
normalization_17_sub_y
normalization_17_sqrt_x
normalization_18_sub_y
normalization_18_sqrt_x
normalization_19_sub_y
normalization_19_sqrt_x

dense_8797: 

dense_8799: 
dense_1_8803: 
dense_1_8805:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCalln
normalization/subSubbattery_powernormalization_sub_y*
T0*'
_output_shapes
:?????????U
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes
:\
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????i
normalization_1/subSubbluenormalization_1_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_1/SqrtSqrtnormalization_1_sqrt_x*
T0*
_output_shapes
:^
normalization_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_1/MaximumMaximumnormalization_1/Sqrt:y:0"normalization_1/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_1/truedivRealDivnormalization_1/sub:z:0normalization_1/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_2/subSubclock_speednormalization_2_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_2/SqrtSqrtnormalization_2_sqrt_x*
T0*
_output_shapes
:^
normalization_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_2/MaximumMaximumnormalization_2/Sqrt:y:0"normalization_2/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_2/truedivRealDivnormalization_2/sub:z:0normalization_2/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_3/subSubdual_simnormalization_3_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_3/SqrtSqrtnormalization_3_sqrt_x*
T0*
_output_shapes
:^
normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*'
_output_shapes
:?????????g
normalization_4/subSubfcnormalization_4_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_4/SqrtSqrtnormalization_4_sqrt_x*
T0*
_output_shapes
:^
normalization_4/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_4/MaximumMaximumnormalization_4/Sqrt:y:0"normalization_4/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_4/truedivRealDivnormalization_4/sub:z:0normalization_4/Maximum:z:0*
T0*'
_output_shapes
:?????????k
normalization_5/subSubfour_gnormalization_5_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_5/SqrtSqrtnormalization_5_sqrt_x*
T0*
_output_shapes
:^
normalization_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_5/MaximumMaximumnormalization_5/Sqrt:y:0"normalization_5/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_5/truedivRealDivnormalization_5/sub:z:0normalization_5/Maximum:z:0*
T0*'
_output_shapes
:?????????o
normalization_6/subSub
int_memorynormalization_6_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_6/SqrtSqrtnormalization_6_sqrt_x*
T0*
_output_shapes
:^
normalization_6/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_6/MaximumMaximumnormalization_6/Sqrt:y:0"normalization_6/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_6/truedivRealDivnormalization_6/sub:z:0normalization_6/Maximum:z:0*
T0*'
_output_shapes
:?????????j
normalization_7/subSubm_depnormalization_7_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_7/SqrtSqrtnormalization_7_sqrt_x*
T0*
_output_shapes
:^
normalization_7/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_7/MaximumMaximumnormalization_7/Sqrt:y:0"normalization_7/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_7/truedivRealDivnormalization_7/sub:z:0normalization_7/Maximum:z:0*
T0*'
_output_shapes
:?????????n
normalization_8/subSub	mobile_wtnormalization_8_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_8/SqrtSqrtnormalization_8_sqrt_x*
T0*
_output_shapes
:^
normalization_8/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_8/MaximumMaximumnormalization_8/Sqrt:y:0"normalization_8/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_8/truedivRealDivnormalization_8/sub:z:0normalization_8/Maximum:z:0*
T0*'
_output_shapes
:?????????l
normalization_9/subSubn_coresnormalization_9_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_9/SqrtSqrtnormalization_9_sqrt_x*
T0*
_output_shapes
:^
normalization_9/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_9/MaximumMaximumnormalization_9/Sqrt:y:0"normalization_9/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_9/truedivRealDivnormalization_9/sub:z:0normalization_9/Maximum:z:0*
T0*'
_output_shapes
:?????????i
normalization_10/subSubpcnormalization_10_sub_y*
T0*'
_output_shapes
:?????????[
normalization_10/SqrtSqrtnormalization_10_sqrt_x*
T0*
_output_shapes
:_
normalization_10/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_10/MaximumMaximumnormalization_10/Sqrt:y:0#normalization_10/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_10/truedivRealDivnormalization_10/sub:z:0normalization_10/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_11/subSub	px_heightnormalization_11_sub_y*
T0*'
_output_shapes
:?????????[
normalization_11/SqrtSqrtnormalization_11_sqrt_x*
T0*
_output_shapes
:_
normalization_11/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_11/MaximumMaximumnormalization_11/Sqrt:y:0#normalization_11/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_11/truedivRealDivnormalization_11/sub:z:0normalization_11/Maximum:z:0*
T0*'
_output_shapes
:?????????o
normalization_12/subSubpx_widthnormalization_12_sub_y*
T0*'
_output_shapes
:?????????[
normalization_12/SqrtSqrtnormalization_12_sqrt_x*
T0*
_output_shapes
:_
normalization_12/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_12/MaximumMaximumnormalization_12/Sqrt:y:0#normalization_12/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_12/truedivRealDivnormalization_12/sub:z:0normalization_12/Maximum:z:0*
T0*'
_output_shapes
:?????????j
normalization_13/subSubramnormalization_13_sub_y*
T0*'
_output_shapes
:?????????[
normalization_13/SqrtSqrtnormalization_13_sqrt_x*
T0*
_output_shapes
:_
normalization_13/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_13/MaximumMaximumnormalization_13/Sqrt:y:0#normalization_13/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_13/truedivRealDivnormalization_13/sub:z:0normalization_13/Maximum:z:0*
T0*'
_output_shapes
:?????????k
normalization_14/subSubsc_hnormalization_14_sub_y*
T0*'
_output_shapes
:?????????[
normalization_14/SqrtSqrtnormalization_14_sqrt_x*
T0*
_output_shapes
:_
normalization_14/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_14/MaximumMaximumnormalization_14/Sqrt:y:0#normalization_14/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_14/truedivRealDivnormalization_14/sub:z:0normalization_14/Maximum:z:0*
T0*'
_output_shapes
:?????????k
normalization_15/subSubsc_wnormalization_15_sub_y*
T0*'
_output_shapes
:?????????[
normalization_15/SqrtSqrtnormalization_15_sqrt_x*
T0*
_output_shapes
:_
normalization_15/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_15/MaximumMaximumnormalization_15/Sqrt:y:0#normalization_15/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_15/truedivRealDivnormalization_15/sub:z:0normalization_15/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_16/subSub	talk_timenormalization_16_sub_y*
T0*'
_output_shapes
:?????????[
normalization_16/SqrtSqrtnormalization_16_sqrt_x*
T0*
_output_shapes
:_
normalization_16/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_16/MaximumMaximumnormalization_16/Sqrt:y:0#normalization_16/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_16/truedivRealDivnormalization_16/sub:z:0normalization_16/Maximum:z:0*
T0*'
_output_shapes
:?????????n
normalization_17/subSubthree_gnormalization_17_sub_y*
T0*'
_output_shapes
:?????????[
normalization_17/SqrtSqrtnormalization_17_sqrt_x*
T0*
_output_shapes
:_
normalization_17/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_17/MaximumMaximumnormalization_17/Sqrt:y:0#normalization_17/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_17/truedivRealDivnormalization_17/sub:z:0normalization_17/Maximum:z:0*
T0*'
_output_shapes
:?????????s
normalization_18/subSubtouch_screennormalization_18_sub_y*
T0*'
_output_shapes
:?????????[
normalization_18/SqrtSqrtnormalization_18_sqrt_x*
T0*
_output_shapes
:_
normalization_18/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_18/MaximumMaximumnormalization_18/Sqrt:y:0#normalization_18/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_18/truedivRealDivnormalization_18/sub:z:0normalization_18/Maximum:z:0*
T0*'
_output_shapes
:?????????k
normalization_19/subSubwifinormalization_19_sub_y*
T0*'
_output_shapes
:?????????[
normalization_19/SqrtSqrtnormalization_19_sqrt_x*
T0*
_output_shapes
:_
normalization_19/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_19/MaximumMaximumnormalization_19/Sqrt:y:0#normalization_19/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_19/truedivRealDivnormalization_19/sub:z:0normalization_19/Maximum:z:0*
T0*'
_output_shapes
:??????????
concatenate/PartitionedCallPartitionedCallnormalization/truediv:z:0normalization_1/truediv:z:0normalization_2/truediv:z:0normalization_3/truediv:z:0normalization_4/truediv:z:0normalization_5/truediv:z:0normalization_6/truediv:z:0normalization_7/truediv:z:0normalization_8/truediv:z:0normalization_9/truediv:z:0normalization_10/truediv:z:0normalization_11/truediv:z:0normalization_12/truediv:z:0normalization_13/truediv:z:0normalization_14/truediv:z:0normalization_15/truediv:z:0normalization_16/truediv:z:0normalization_17/truediv:z:0normalization_18/truediv:z:0normalization_19/truediv:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_7920?
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0
dense_8797
dense_8799*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_7933?
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_7944?
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_8803dense_1_8805*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_7957w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????::::::::::::::::::::::::::::::::::::::::: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namebattery_power:MI
'
_output_shapes
:?????????

_user_specified_nameblue:TP
'
_output_shapes
:?????????
%
_user_specified_nameclock_speed:QM
'
_output_shapes
:?????????
"
_user_specified_name
dual_sim:KG
'
_output_shapes
:?????????

_user_specified_namefc:OK
'
_output_shapes
:?????????
 
_user_specified_namefour_g:SO
'
_output_shapes
:?????????
$
_user_specified_name
int_memory:NJ
'
_output_shapes
:?????????

_user_specified_namem_dep:RN
'
_output_shapes
:?????????
#
_user_specified_name	mobile_wt:P	L
'
_output_shapes
:?????????
!
_user_specified_name	n_cores:K
G
'
_output_shapes
:?????????

_user_specified_namepc:RN
'
_output_shapes
:?????????
#
_user_specified_name	px_height:QM
'
_output_shapes
:?????????
"
_user_specified_name
px_width:LH
'
_output_shapes
:?????????

_user_specified_nameram:MI
'
_output_shapes
:?????????

_user_specified_namesc_h:MI
'
_output_shapes
:?????????

_user_specified_namesc_w:RN
'
_output_shapes
:?????????
#
_user_specified_name	talk_time:PL
'
_output_shapes
:?????????
!
_user_specified_name	three_g:UQ
'
_output_shapes
:?????????
&
_user_specified_nametouch_screen:MI
'
_output_shapes
:?????????

_user_specified_namewifi: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::  

_output_shapes
:: !

_output_shapes
:: "

_output_shapes
:: #

_output_shapes
:: $

_output_shapes
:: %

_output_shapes
:: &

_output_shapes
:: '

_output_shapes
:: (

_output_shapes
:: )

_output_shapes
:: *

_output_shapes
:: +

_output_shapes
:: ,

_output_shapes
:: -

_output_shapes
:: .

_output_shapes
:: /

_output_shapes
:: 0

_output_shapes
:: 1

_output_shapes
:: 2

_output_shapes
:: 3

_output_shapes
:: 4

_output_shapes
:: 5

_output_shapes
:: 6

_output_shapes
:: 7

_output_shapes
:: 8

_output_shapes
:: 9

_output_shapes
:: :

_output_shapes
:: ;

_output_shapes
:
?
?
%__inference_dense_layer_call_fn_10685

inputs
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_7933o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?/
?
"__inference_signature_wrapper_9098
battery_power
blue
clock_speed
dual_sim
fc

four_g

int_memory	
m_dep
	mobile_wt
n_cores
pc
	px_height
px_width
ram
sc_h
sc_w
	talk_time
three_g
touch_screen
wifi
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39: 

unknown_40: 

unknown_41: 

unknown_42:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallbattery_powerblueclock_speeddual_simfcfour_g
int_memorym_dep	mobile_wtn_corespc	px_heightpx_widthramsc_hsc_w	talk_timethree_gtouch_screenwifiunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*K
TinD
B2@*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
<=>?*-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__wrapped_model_7710o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????::::::::::::::::::::::::::::::::::::::::: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namebattery_power:MI
'
_output_shapes
:?????????

_user_specified_nameblue:TP
'
_output_shapes
:?????????
%
_user_specified_nameclock_speed:QM
'
_output_shapes
:?????????
"
_user_specified_name
dual_sim:KG
'
_output_shapes
:?????????

_user_specified_namefc:OK
'
_output_shapes
:?????????
 
_user_specified_namefour_g:SO
'
_output_shapes
:?????????
$
_user_specified_name
int_memory:NJ
'
_output_shapes
:?????????

_user_specified_namem_dep:RN
'
_output_shapes
:?????????
#
_user_specified_name	mobile_wt:P	L
'
_output_shapes
:?????????
!
_user_specified_name	n_cores:K
G
'
_output_shapes
:?????????

_user_specified_namepc:RN
'
_output_shapes
:?????????
#
_user_specified_name	px_height:QM
'
_output_shapes
:?????????
"
_user_specified_name
px_width:LH
'
_output_shapes
:?????????

_user_specified_nameram:MI
'
_output_shapes
:?????????

_user_specified_namesc_h:MI
'
_output_shapes
:?????????

_user_specified_namesc_w:RN
'
_output_shapes
:?????????
#
_user_specified_name	talk_time:PL
'
_output_shapes
:?????????
!
_user_specified_name	three_g:UQ
'
_output_shapes
:?????????
&
_user_specified_nametouch_screen:MI
'
_output_shapes
:?????????

_user_specified_namewifi: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::  

_output_shapes
:: !

_output_shapes
:: "

_output_shapes
:: #

_output_shapes
:: $

_output_shapes
:: %

_output_shapes
:: &

_output_shapes
:: '

_output_shapes
:: (

_output_shapes
:: )

_output_shapes
:: *

_output_shapes
:: +

_output_shapes
:: ,

_output_shapes
:: -

_output_shapes
:: .

_output_shapes
:: /

_output_shapes
:: 0

_output_shapes
:: 1

_output_shapes
:: 2

_output_shapes
:: 3

_output_shapes
:: 4

_output_shapes
:: 5

_output_shapes
:: 6

_output_shapes
:: 7

_output_shapes
:: 8

_output_shapes
:: 9

_output_shapes
:: :

_output_shapes
:: ;

_output_shapes
:
??
?!
!__inference__traced_restore_11243
file_prefix
assignvariableop_mean: %
assignvariableop_1_variance: "
assignvariableop_2_count:	 #
assignvariableop_3_mean_1: '
assignvariableop_4_variance_1: $
assignvariableop_5_count_1:	 #
assignvariableop_6_mean_2: '
assignvariableop_7_variance_2: $
assignvariableop_8_count_2:	 #
assignvariableop_9_mean_3: (
assignvariableop_10_variance_3: %
assignvariableop_11_count_3:	 $
assignvariableop_12_mean_4: (
assignvariableop_13_variance_4: %
assignvariableop_14_count_4:	 $
assignvariableop_15_mean_5: (
assignvariableop_16_variance_5: %
assignvariableop_17_count_5:	 $
assignvariableop_18_mean_6: (
assignvariableop_19_variance_6: %
assignvariableop_20_count_6:	 $
assignvariableop_21_mean_7: (
assignvariableop_22_variance_7: %
assignvariableop_23_count_7:	 $
assignvariableop_24_mean_8: (
assignvariableop_25_variance_8: %
assignvariableop_26_count_8:	 $
assignvariableop_27_mean_9: (
assignvariableop_28_variance_9: %
assignvariableop_29_count_9:	 %
assignvariableop_30_mean_10: )
assignvariableop_31_variance_10: &
assignvariableop_32_count_10:	 %
assignvariableop_33_mean_11: )
assignvariableop_34_variance_11: &
assignvariableop_35_count_11:	 %
assignvariableop_36_mean_12: )
assignvariableop_37_variance_12: &
assignvariableop_38_count_12:	 %
assignvariableop_39_mean_13: )
assignvariableop_40_variance_13: &
assignvariableop_41_count_13:	 %
assignvariableop_42_mean_14: )
assignvariableop_43_variance_14: &
assignvariableop_44_count_14:	 %
assignvariableop_45_mean_15: )
assignvariableop_46_variance_15: &
assignvariableop_47_count_15:	 %
assignvariableop_48_mean_16: )
assignvariableop_49_variance_16: &
assignvariableop_50_count_16:	 %
assignvariableop_51_mean_17: )
assignvariableop_52_variance_17: &
assignvariableop_53_count_17:	 %
assignvariableop_54_mean_18: )
assignvariableop_55_variance_18: &
assignvariableop_56_count_18:	 %
assignvariableop_57_mean_19: )
assignvariableop_58_variance_19: &
assignvariableop_59_count_19:	 2
 assignvariableop_60_dense_kernel: ,
assignvariableop_61_dense_bias: 4
"assignvariableop_62_dense_1_kernel: .
 assignvariableop_63_dense_1_bias:#
assignvariableop_64_total: &
assignvariableop_65_count_20: %
assignvariableop_66_total_1: &
assignvariableop_67_count_21: 
identity_69??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:E*
dtype0*?
value?B?EB4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-1/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-2/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-3/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-4/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-5/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-6/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-7/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-8/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-9/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/count/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/mean/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-10/variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/count/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/mean/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-11/variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/count/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/mean/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-12/variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/count/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/mean/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-13/variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/count/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/mean/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-14/variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/count/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/mean/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-15/variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/count/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/mean/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-16/variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/count/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/mean/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-17/variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/count/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/mean/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-18/variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-18/count/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/mean/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-19/variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-19/count/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:E*
dtype0*?
value?B?EB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*S
dtypesI
G2E																				[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_meanIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_varianceIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_countIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_mean_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_variance_1Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_count_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_mean_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_variance_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_count_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_mean_3Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_variance_3Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_3Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_mean_4Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_variance_4Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_4Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_mean_5Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpassignvariableop_16_variance_5Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpassignvariableop_17_count_5Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpassignvariableop_18_mean_6Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpassignvariableop_19_variance_6Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_6Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpassignvariableop_21_mean_7Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOpassignvariableop_22_variance_7Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_23AssignVariableOpassignvariableop_23_count_7Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOpassignvariableop_24_mean_8Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOpassignvariableop_25_variance_8Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpassignvariableop_26_count_8Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOpassignvariableop_27_mean_9Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOpassignvariableop_28_variance_9Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_29AssignVariableOpassignvariableop_29_count_9Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOpassignvariableop_30_mean_10Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOpassignvariableop_31_variance_10Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_32AssignVariableOpassignvariableop_32_count_10Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOpassignvariableop_33_mean_11Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOpassignvariableop_34_variance_11Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_35AssignVariableOpassignvariableop_35_count_11Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOpassignvariableop_36_mean_12Identity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOpassignvariableop_37_variance_12Identity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_38AssignVariableOpassignvariableop_38_count_12Identity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOpassignvariableop_39_mean_13Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOpassignvariableop_40_variance_13Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_41AssignVariableOpassignvariableop_41_count_13Identity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOpassignvariableop_42_mean_14Identity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOpassignvariableop_43_variance_14Identity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_44AssignVariableOpassignvariableop_44_count_14Identity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOpassignvariableop_45_mean_15Identity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOpassignvariableop_46_variance_15Identity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_47AssignVariableOpassignvariableop_47_count_15Identity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOpassignvariableop_48_mean_16Identity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOpassignvariableop_49_variance_16Identity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_50AssignVariableOpassignvariableop_50_count_16Identity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOpassignvariableop_51_mean_17Identity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOpassignvariableop_52_variance_17Identity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_53AssignVariableOpassignvariableop_53_count_17Identity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOpassignvariableop_54_mean_18Identity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOpassignvariableop_55_variance_18Identity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_56AssignVariableOpassignvariableop_56_count_18Identity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOpassignvariableop_57_mean_19Identity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOpassignvariableop_58_variance_19Identity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_59AssignVariableOpassignvariableop_59_count_19Identity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOp assignvariableop_60_dense_kernelIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOpassignvariableop_61_dense_biasIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOp"assignvariableop_62_dense_1_kernelIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOp assignvariableop_63_dense_1_biasIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOpassignvariableop_64_totalIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOpassignvariableop_65_count_20Identity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOpassignvariableop_66_total_1Identity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOpassignvariableop_67_count_21Identity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_68Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_69IdentityIdentity_68:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_69Identity_69:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
??
?
?__inference_model_layer_call_and_return_conditional_losses_9689
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
	inputs_13
	inputs_14
	inputs_15
	inputs_16
	inputs_17
	inputs_18
	inputs_19
normalization_sub_y
normalization_sqrt_x
normalization_1_sub_y
normalization_1_sqrt_x
normalization_2_sub_y
normalization_2_sqrt_x
normalization_3_sub_y
normalization_3_sqrt_x
normalization_4_sub_y
normalization_4_sqrt_x
normalization_5_sub_y
normalization_5_sqrt_x
normalization_6_sub_y
normalization_6_sqrt_x
normalization_7_sub_y
normalization_7_sqrt_x
normalization_8_sub_y
normalization_8_sqrt_x
normalization_9_sub_y
normalization_9_sqrt_x
normalization_10_sub_y
normalization_10_sqrt_x
normalization_11_sub_y
normalization_11_sqrt_x
normalization_12_sub_y
normalization_12_sqrt_x
normalization_13_sub_y
normalization_13_sqrt_x
normalization_14_sub_y
normalization_14_sqrt_x
normalization_15_sub_y
normalization_15_sqrt_x
normalization_16_sub_y
normalization_16_sqrt_x
normalization_17_sub_y
normalization_17_sqrt_x
normalization_18_sub_y
normalization_18_sqrt_x
normalization_19_sub_y
normalization_19_sqrt_x6
$dense_matmul_readvariableop_resource: 3
%dense_biasadd_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOpi
normalization/subSubinputs_0normalization_sub_y*
T0*'
_output_shapes
:?????????U
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes
:\
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_1/subSubinputs_1normalization_1_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_1/SqrtSqrtnormalization_1_sqrt_x*
T0*
_output_shapes
:^
normalization_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_1/MaximumMaximumnormalization_1/Sqrt:y:0"normalization_1/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_1/truedivRealDivnormalization_1/sub:z:0normalization_1/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_2/subSubinputs_2normalization_2_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_2/SqrtSqrtnormalization_2_sqrt_x*
T0*
_output_shapes
:^
normalization_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_2/MaximumMaximumnormalization_2/Sqrt:y:0"normalization_2/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_2/truedivRealDivnormalization_2/sub:z:0normalization_2/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_3/subSubinputs_3normalization_3_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_3/SqrtSqrtnormalization_3_sqrt_x*
T0*
_output_shapes
:^
normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_4/subSubinputs_4normalization_4_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_4/SqrtSqrtnormalization_4_sqrt_x*
T0*
_output_shapes
:^
normalization_4/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_4/MaximumMaximumnormalization_4/Sqrt:y:0"normalization_4/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_4/truedivRealDivnormalization_4/sub:z:0normalization_4/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_5/subSubinputs_5normalization_5_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_5/SqrtSqrtnormalization_5_sqrt_x*
T0*
_output_shapes
:^
normalization_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_5/MaximumMaximumnormalization_5/Sqrt:y:0"normalization_5/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_5/truedivRealDivnormalization_5/sub:z:0normalization_5/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_6/subSubinputs_6normalization_6_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_6/SqrtSqrtnormalization_6_sqrt_x*
T0*
_output_shapes
:^
normalization_6/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_6/MaximumMaximumnormalization_6/Sqrt:y:0"normalization_6/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_6/truedivRealDivnormalization_6/sub:z:0normalization_6/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_7/subSubinputs_7normalization_7_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_7/SqrtSqrtnormalization_7_sqrt_x*
T0*
_output_shapes
:^
normalization_7/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_7/MaximumMaximumnormalization_7/Sqrt:y:0"normalization_7/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_7/truedivRealDivnormalization_7/sub:z:0normalization_7/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_8/subSubinputs_8normalization_8_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_8/SqrtSqrtnormalization_8_sqrt_x*
T0*
_output_shapes
:^
normalization_8/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_8/MaximumMaximumnormalization_8/Sqrt:y:0"normalization_8/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_8/truedivRealDivnormalization_8/sub:z:0normalization_8/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_9/subSubinputs_9normalization_9_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_9/SqrtSqrtnormalization_9_sqrt_x*
T0*
_output_shapes
:^
normalization_9/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_9/MaximumMaximumnormalization_9/Sqrt:y:0"normalization_9/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_9/truedivRealDivnormalization_9/sub:z:0normalization_9/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_10/subSub	inputs_10normalization_10_sub_y*
T0*'
_output_shapes
:?????????[
normalization_10/SqrtSqrtnormalization_10_sqrt_x*
T0*
_output_shapes
:_
normalization_10/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_10/MaximumMaximumnormalization_10/Sqrt:y:0#normalization_10/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_10/truedivRealDivnormalization_10/sub:z:0normalization_10/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_11/subSub	inputs_11normalization_11_sub_y*
T0*'
_output_shapes
:?????????[
normalization_11/SqrtSqrtnormalization_11_sqrt_x*
T0*
_output_shapes
:_
normalization_11/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_11/MaximumMaximumnormalization_11/Sqrt:y:0#normalization_11/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_11/truedivRealDivnormalization_11/sub:z:0normalization_11/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_12/subSub	inputs_12normalization_12_sub_y*
T0*'
_output_shapes
:?????????[
normalization_12/SqrtSqrtnormalization_12_sqrt_x*
T0*
_output_shapes
:_
normalization_12/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_12/MaximumMaximumnormalization_12/Sqrt:y:0#normalization_12/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_12/truedivRealDivnormalization_12/sub:z:0normalization_12/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_13/subSub	inputs_13normalization_13_sub_y*
T0*'
_output_shapes
:?????????[
normalization_13/SqrtSqrtnormalization_13_sqrt_x*
T0*
_output_shapes
:_
normalization_13/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_13/MaximumMaximumnormalization_13/Sqrt:y:0#normalization_13/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_13/truedivRealDivnormalization_13/sub:z:0normalization_13/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_14/subSub	inputs_14normalization_14_sub_y*
T0*'
_output_shapes
:?????????[
normalization_14/SqrtSqrtnormalization_14_sqrt_x*
T0*
_output_shapes
:_
normalization_14/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_14/MaximumMaximumnormalization_14/Sqrt:y:0#normalization_14/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_14/truedivRealDivnormalization_14/sub:z:0normalization_14/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_15/subSub	inputs_15normalization_15_sub_y*
T0*'
_output_shapes
:?????????[
normalization_15/SqrtSqrtnormalization_15_sqrt_x*
T0*
_output_shapes
:_
normalization_15/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_15/MaximumMaximumnormalization_15/Sqrt:y:0#normalization_15/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_15/truedivRealDivnormalization_15/sub:z:0normalization_15/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_16/subSub	inputs_16normalization_16_sub_y*
T0*'
_output_shapes
:?????????[
normalization_16/SqrtSqrtnormalization_16_sqrt_x*
T0*
_output_shapes
:_
normalization_16/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_16/MaximumMaximumnormalization_16/Sqrt:y:0#normalization_16/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_16/truedivRealDivnormalization_16/sub:z:0normalization_16/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_17/subSub	inputs_17normalization_17_sub_y*
T0*'
_output_shapes
:?????????[
normalization_17/SqrtSqrtnormalization_17_sqrt_x*
T0*
_output_shapes
:_
normalization_17/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_17/MaximumMaximumnormalization_17/Sqrt:y:0#normalization_17/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_17/truedivRealDivnormalization_17/sub:z:0normalization_17/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_18/subSub	inputs_18normalization_18_sub_y*
T0*'
_output_shapes
:?????????[
normalization_18/SqrtSqrtnormalization_18_sqrt_x*
T0*
_output_shapes
:_
normalization_18/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_18/MaximumMaximumnormalization_18/Sqrt:y:0#normalization_18/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_18/truedivRealDivnormalization_18/sub:z:0normalization_18/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_19/subSub	inputs_19normalization_19_sub_y*
T0*'
_output_shapes
:?????????[
normalization_19/SqrtSqrtnormalization_19_sqrt_x*
T0*
_output_shapes
:_
normalization_19/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_19/MaximumMaximumnormalization_19/Sqrt:y:0#normalization_19/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_19/truedivRealDivnormalization_19/sub:z:0normalization_19/Maximum:z:0*
T0*'
_output_shapes
:?????????Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate/concatConcatV2normalization/truediv:z:0normalization_1/truediv:z:0normalization_2/truediv:z:0normalization_3/truediv:z:0normalization_4/truediv:z:0normalization_5/truediv:z:0normalization_6/truediv:z:0normalization_7/truediv:z:0normalization_8/truediv:z:0normalization_9/truediv:z:0normalization_10/truediv:z:0normalization_11/truediv:z:0normalization_12/truediv:z:0normalization_13/truediv:z:0normalization_14/truediv:z:0normalization_15/truediv:z:0normalization_16/truediv:z:0normalization_17/truediv:z:0normalization_18/truediv:z:0normalization_19/truediv:z:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:??????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? \

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:????????? Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
dropout/dropout/MulMuldense/Relu:activations:0dropout/dropout/Const:output:0*
T0*'
_output_shapes
:????????? ]
dropout/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? ?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:????????? ?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????b
IdentityIdentitydense_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????::::::::::::::::::::::::::::::::::::::::: : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/8:Q	M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/9:R
N
'
_output_shapes
:?????????
#
_user_specified_name	inputs/10:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/11:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/12:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/13:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/14:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/15:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/16:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/17:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/18:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/19: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::  

_output_shapes
:: !

_output_shapes
:: "

_output_shapes
:: #

_output_shapes
:: $

_output_shapes
:: %

_output_shapes
:: &

_output_shapes
:: '

_output_shapes
:: (

_output_shapes
:: )

_output_shapes
:: *

_output_shapes
:: +

_output_shapes
:: ,

_output_shapes
:: -

_output_shapes
:: .

_output_shapes
:: /

_output_shapes
:: 0

_output_shapes
:: 1

_output_shapes
:: 2

_output_shapes
:: 3

_output_shapes
:: 4

_output_shapes
:: 5

_output_shapes
:: 6

_output_shapes
:: 7

_output_shapes
:: 8

_output_shapes
:: 9

_output_shapes
:: :

_output_shapes
:: ;

_output_shapes
:
?'
?
__inference_adapt_step_10017
iterator

iterator_1%
add_readvariableop_resource:	 !
readvariableop_resource: #
readvariableop_2_resource: ??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:?????????*"
output_shapes
:?????????*
output_types
2g
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*#
_output_shapes
:?????????h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: }
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(`
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:?
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*#
_output_shapes
:?????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(i
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 o
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 M
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	Z
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:O
ConstConst*
_output_shapes
:*
dtype0*
valueB: P
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: f
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	X
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: M
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: G
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: K
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: ^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
: T
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
: C
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
: `
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0R
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
: J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @F
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
: b
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype0R
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
: A
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
: R
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
: L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
: V
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
: E
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
: E
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: ?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype0?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
??
?
?__inference_model_layer_call_and_return_conditional_losses_9502
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
	inputs_13
	inputs_14
	inputs_15
	inputs_16
	inputs_17
	inputs_18
	inputs_19
normalization_sub_y
normalization_sqrt_x
normalization_1_sub_y
normalization_1_sqrt_x
normalization_2_sub_y
normalization_2_sqrt_x
normalization_3_sub_y
normalization_3_sqrt_x
normalization_4_sub_y
normalization_4_sqrt_x
normalization_5_sub_y
normalization_5_sqrt_x
normalization_6_sub_y
normalization_6_sqrt_x
normalization_7_sub_y
normalization_7_sqrt_x
normalization_8_sub_y
normalization_8_sqrt_x
normalization_9_sub_y
normalization_9_sqrt_x
normalization_10_sub_y
normalization_10_sqrt_x
normalization_11_sub_y
normalization_11_sqrt_x
normalization_12_sub_y
normalization_12_sqrt_x
normalization_13_sub_y
normalization_13_sqrt_x
normalization_14_sub_y
normalization_14_sqrt_x
normalization_15_sub_y
normalization_15_sqrt_x
normalization_16_sub_y
normalization_16_sqrt_x
normalization_17_sub_y
normalization_17_sqrt_x
normalization_18_sub_y
normalization_18_sqrt_x
normalization_19_sub_y
normalization_19_sqrt_x6
$dense_matmul_readvariableop_resource: 3
%dense_biasadd_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOpi
normalization/subSubinputs_0normalization_sub_y*
T0*'
_output_shapes
:?????????U
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes
:\
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_1/subSubinputs_1normalization_1_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_1/SqrtSqrtnormalization_1_sqrt_x*
T0*
_output_shapes
:^
normalization_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_1/MaximumMaximumnormalization_1/Sqrt:y:0"normalization_1/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_1/truedivRealDivnormalization_1/sub:z:0normalization_1/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_2/subSubinputs_2normalization_2_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_2/SqrtSqrtnormalization_2_sqrt_x*
T0*
_output_shapes
:^
normalization_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_2/MaximumMaximumnormalization_2/Sqrt:y:0"normalization_2/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_2/truedivRealDivnormalization_2/sub:z:0normalization_2/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_3/subSubinputs_3normalization_3_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_3/SqrtSqrtnormalization_3_sqrt_x*
T0*
_output_shapes
:^
normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_4/subSubinputs_4normalization_4_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_4/SqrtSqrtnormalization_4_sqrt_x*
T0*
_output_shapes
:^
normalization_4/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_4/MaximumMaximumnormalization_4/Sqrt:y:0"normalization_4/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_4/truedivRealDivnormalization_4/sub:z:0normalization_4/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_5/subSubinputs_5normalization_5_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_5/SqrtSqrtnormalization_5_sqrt_x*
T0*
_output_shapes
:^
normalization_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_5/MaximumMaximumnormalization_5/Sqrt:y:0"normalization_5/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_5/truedivRealDivnormalization_5/sub:z:0normalization_5/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_6/subSubinputs_6normalization_6_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_6/SqrtSqrtnormalization_6_sqrt_x*
T0*
_output_shapes
:^
normalization_6/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_6/MaximumMaximumnormalization_6/Sqrt:y:0"normalization_6/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_6/truedivRealDivnormalization_6/sub:z:0normalization_6/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_7/subSubinputs_7normalization_7_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_7/SqrtSqrtnormalization_7_sqrt_x*
T0*
_output_shapes
:^
normalization_7/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_7/MaximumMaximumnormalization_7/Sqrt:y:0"normalization_7/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_7/truedivRealDivnormalization_7/sub:z:0normalization_7/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_8/subSubinputs_8normalization_8_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_8/SqrtSqrtnormalization_8_sqrt_x*
T0*
_output_shapes
:^
normalization_8/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_8/MaximumMaximumnormalization_8/Sqrt:y:0"normalization_8/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_8/truedivRealDivnormalization_8/sub:z:0normalization_8/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_9/subSubinputs_9normalization_9_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_9/SqrtSqrtnormalization_9_sqrt_x*
T0*
_output_shapes
:^
normalization_9/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_9/MaximumMaximumnormalization_9/Sqrt:y:0"normalization_9/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_9/truedivRealDivnormalization_9/sub:z:0normalization_9/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_10/subSub	inputs_10normalization_10_sub_y*
T0*'
_output_shapes
:?????????[
normalization_10/SqrtSqrtnormalization_10_sqrt_x*
T0*
_output_shapes
:_
normalization_10/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_10/MaximumMaximumnormalization_10/Sqrt:y:0#normalization_10/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_10/truedivRealDivnormalization_10/sub:z:0normalization_10/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_11/subSub	inputs_11normalization_11_sub_y*
T0*'
_output_shapes
:?????????[
normalization_11/SqrtSqrtnormalization_11_sqrt_x*
T0*
_output_shapes
:_
normalization_11/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_11/MaximumMaximumnormalization_11/Sqrt:y:0#normalization_11/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_11/truedivRealDivnormalization_11/sub:z:0normalization_11/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_12/subSub	inputs_12normalization_12_sub_y*
T0*'
_output_shapes
:?????????[
normalization_12/SqrtSqrtnormalization_12_sqrt_x*
T0*
_output_shapes
:_
normalization_12/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_12/MaximumMaximumnormalization_12/Sqrt:y:0#normalization_12/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_12/truedivRealDivnormalization_12/sub:z:0normalization_12/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_13/subSub	inputs_13normalization_13_sub_y*
T0*'
_output_shapes
:?????????[
normalization_13/SqrtSqrtnormalization_13_sqrt_x*
T0*
_output_shapes
:_
normalization_13/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_13/MaximumMaximumnormalization_13/Sqrt:y:0#normalization_13/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_13/truedivRealDivnormalization_13/sub:z:0normalization_13/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_14/subSub	inputs_14normalization_14_sub_y*
T0*'
_output_shapes
:?????????[
normalization_14/SqrtSqrtnormalization_14_sqrt_x*
T0*
_output_shapes
:_
normalization_14/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_14/MaximumMaximumnormalization_14/Sqrt:y:0#normalization_14/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_14/truedivRealDivnormalization_14/sub:z:0normalization_14/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_15/subSub	inputs_15normalization_15_sub_y*
T0*'
_output_shapes
:?????????[
normalization_15/SqrtSqrtnormalization_15_sqrt_x*
T0*
_output_shapes
:_
normalization_15/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_15/MaximumMaximumnormalization_15/Sqrt:y:0#normalization_15/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_15/truedivRealDivnormalization_15/sub:z:0normalization_15/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_16/subSub	inputs_16normalization_16_sub_y*
T0*'
_output_shapes
:?????????[
normalization_16/SqrtSqrtnormalization_16_sqrt_x*
T0*
_output_shapes
:_
normalization_16/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_16/MaximumMaximumnormalization_16/Sqrt:y:0#normalization_16/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_16/truedivRealDivnormalization_16/sub:z:0normalization_16/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_17/subSub	inputs_17normalization_17_sub_y*
T0*'
_output_shapes
:?????????[
normalization_17/SqrtSqrtnormalization_17_sqrt_x*
T0*
_output_shapes
:_
normalization_17/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_17/MaximumMaximumnormalization_17/Sqrt:y:0#normalization_17/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_17/truedivRealDivnormalization_17/sub:z:0normalization_17/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_18/subSub	inputs_18normalization_18_sub_y*
T0*'
_output_shapes
:?????????[
normalization_18/SqrtSqrtnormalization_18_sqrt_x*
T0*
_output_shapes
:_
normalization_18/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_18/MaximumMaximumnormalization_18/Sqrt:y:0#normalization_18/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_18/truedivRealDivnormalization_18/sub:z:0normalization_18/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_19/subSub	inputs_19normalization_19_sub_y*
T0*'
_output_shapes
:?????????[
normalization_19/SqrtSqrtnormalization_19_sqrt_x*
T0*
_output_shapes
:_
normalization_19/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_19/MaximumMaximumnormalization_19/Sqrt:y:0#normalization_19/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_19/truedivRealDivnormalization_19/sub:z:0normalization_19/Maximum:z:0*
T0*'
_output_shapes
:?????????Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate/concatConcatV2normalization/truediv:z:0normalization_1/truediv:z:0normalization_2/truediv:z:0normalization_3/truediv:z:0normalization_4/truediv:z:0normalization_5/truediv:z:0normalization_6/truediv:z:0normalization_7/truediv:z:0normalization_8/truediv:z:0normalization_9/truediv:z:0normalization_10/truediv:z:0normalization_11/truediv:z:0normalization_12/truediv:z:0normalization_13/truediv:z:0normalization_14/truediv:z:0normalization_15/truediv:z:0normalization_16/truediv:z:0normalization_17/truediv:z:0normalization_18/truediv:z:0normalization_19/truediv:z:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:??????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? \

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:????????? h
dropout/IdentityIdentitydense/Relu:activations:0*
T0*'
_output_shapes
:????????? ?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????b
IdentityIdentitydense_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????::::::::::::::::::::::::::::::::::::::::: : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/8:Q	M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/9:R
N
'
_output_shapes
:?????????
#
_user_specified_name	inputs/10:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/11:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/12:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/13:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/14:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/15:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/16:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/17:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/18:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/19: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::  

_output_shapes
:: !

_output_shapes
:: "

_output_shapes
:: #

_output_shapes
:: $

_output_shapes
:: %

_output_shapes
:: &

_output_shapes
:: '

_output_shapes
:: (

_output_shapes
:: )

_output_shapes
:: *

_output_shapes
:: +

_output_shapes
:: ,

_output_shapes
:: -

_output_shapes
:: .

_output_shapes
:: /

_output_shapes
:: 0

_output_shapes
:: 1

_output_shapes
:: 2

_output_shapes
:: 3

_output_shapes
:: 4

_output_shapes
:: 5

_output_shapes
:: 6

_output_shapes
:: 7

_output_shapes
:: 8

_output_shapes
:: 9

_output_shapes
:: :

_output_shapes
:: ;

_output_shapes
:
?'
?
__inference_adapt_step_9923
iterator

iterator_1%
add_readvariableop_resource:	 !
readvariableop_resource: #
readvariableop_2_resource: ??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:?????????*"
output_shapes
:?????????*
output_types
2g
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*#
_output_shapes
:?????????h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: }
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(`
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:?
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*#
_output_shapes
:?????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(i
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 o
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 M
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	Z
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:O
ConstConst*
_output_shapes
:*
dtype0*
valueB: P
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: f
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	X
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: M
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: G
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: K
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: ^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
: T
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
: C
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
: `
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0R
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
: J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @F
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
: b
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype0R
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
: A
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
: R
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
: L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
: V
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
: E
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
: E
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: ?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype0?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
?'
?
__inference_adapt_step_10439
iterator

iterator_1%
add_readvariableop_resource:	 !
readvariableop_resource: #
readvariableop_2_resource: ??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:?????????*"
output_shapes
:?????????*
output_types
2g
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*#
_output_shapes
:?????????h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: }
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(`
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:?
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*#
_output_shapes
:?????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(i
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 o
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 M
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	Z
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:O
ConstConst*
_output_shapes
:*
dtype0*
valueB: P
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: f
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	X
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: M
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: G
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: K
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: ^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
: T
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
: C
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
: `
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0R
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
: J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @F
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
: b
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype0R
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
: A
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
: R
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
: L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
: V
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
: E
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
: E
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: ?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype0?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
?0
?
$__inference_model_layer_call_fn_9322
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
	inputs_13
	inputs_14
	inputs_15
	inputs_16
	inputs_17
	inputs_18
	inputs_19
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39: 

unknown_40: 

unknown_41: 

unknown_42:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12	inputs_13	inputs_14	inputs_15	inputs_16	inputs_17	inputs_18	inputs_19unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*K
TinD
B2@*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
<=>?*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_8431o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????::::::::::::::::::::::::::::::::::::::::: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/8:Q	M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/9:R
N
'
_output_shapes
:?????????
#
_user_specified_name	inputs/10:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/11:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/12:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/13:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/14:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/15:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/16:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/17:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/18:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/19: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::  

_output_shapes
:: !

_output_shapes
:: "

_output_shapes
:: #

_output_shapes
:: $

_output_shapes
:: %

_output_shapes
:: &

_output_shapes
:: '

_output_shapes
:: (

_output_shapes
:: )

_output_shapes
:: *

_output_shapes
:: +

_output_shapes
:: ,

_output_shapes
:: -

_output_shapes
:: .

_output_shapes
:: /

_output_shapes
:: 0

_output_shapes
:: 1

_output_shapes
:: 2

_output_shapes
:: 3

_output_shapes
:: 4

_output_shapes
:: 5

_output_shapes
:: 6

_output_shapes
:: 7

_output_shapes
:: 8

_output_shapes
:: 9

_output_shapes
:: :

_output_shapes
:: ;

_output_shapes
:
?'
?
__inference_adapt_step_10157
iterator

iterator_1%
add_readvariableop_resource:	 !
readvariableop_resource: #
readvariableop_2_resource: ??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:?????????*"
output_shapes
:?????????*
output_types
2g
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*#
_output_shapes
:?????????h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: }
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(`
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:?
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*#
_output_shapes
:?????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(i
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 o
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 M
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	Z
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:O
ConstConst*
_output_shapes
:*
dtype0*
valueB: P
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: f
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	X
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: M
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: G
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: K
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: ^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
: T
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
: C
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
: `
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0R
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
: J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @F
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
: b
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype0R
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
: A
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
: R
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
: L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
: V
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
: E
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
: E
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: ?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype0?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
?&
?
__inference_adapt_step_9829
iterator

iterator_1%
add_readvariableop_resource:	 !
readvariableop_resource: #
readvariableop_2_resource: ??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:?????????*"
output_shapes
:?????????*
output_types
2h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/meanMeanIteratorGetNext:components:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(`
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:?
moments/SquaredDifferenceSquaredDifferenceIteratorGetNext:components:0moments/StopGradient:output:0*
T0*#
_output_shapes
:?????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(i
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 o
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 a
ShapeShapeIteratorGetNext:components:0*
T0*
_output_shapes
:*
out_type0	Z
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:O
ConstConst*
_output_shapes
:*
dtype0*
valueB: P
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: f
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	X
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: K
CastCastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: G
Cast_1Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: I
truedivRealDivCast:y:0
Cast_1:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: ^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
: T
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
: C
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
: `
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0R
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
: J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @F
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
: b
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype0R
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
: A
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
: R
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
: L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
: V
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
: E
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
: E
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: ?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype0?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
?k
?
__inference__traced_save_11029
file_prefix#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop$
 savev2_count_read_readvariableop	%
!savev2_mean_1_read_readvariableop)
%savev2_variance_1_read_readvariableop&
"savev2_count_1_read_readvariableop	%
!savev2_mean_2_read_readvariableop)
%savev2_variance_2_read_readvariableop&
"savev2_count_2_read_readvariableop	%
!savev2_mean_3_read_readvariableop)
%savev2_variance_3_read_readvariableop&
"savev2_count_3_read_readvariableop	%
!savev2_mean_4_read_readvariableop)
%savev2_variance_4_read_readvariableop&
"savev2_count_4_read_readvariableop	%
!savev2_mean_5_read_readvariableop)
%savev2_variance_5_read_readvariableop&
"savev2_count_5_read_readvariableop	%
!savev2_mean_6_read_readvariableop)
%savev2_variance_6_read_readvariableop&
"savev2_count_6_read_readvariableop	%
!savev2_mean_7_read_readvariableop)
%savev2_variance_7_read_readvariableop&
"savev2_count_7_read_readvariableop	%
!savev2_mean_8_read_readvariableop)
%savev2_variance_8_read_readvariableop&
"savev2_count_8_read_readvariableop	%
!savev2_mean_9_read_readvariableop)
%savev2_variance_9_read_readvariableop&
"savev2_count_9_read_readvariableop	&
"savev2_mean_10_read_readvariableop*
&savev2_variance_10_read_readvariableop'
#savev2_count_10_read_readvariableop	&
"savev2_mean_11_read_readvariableop*
&savev2_variance_11_read_readvariableop'
#savev2_count_11_read_readvariableop	&
"savev2_mean_12_read_readvariableop*
&savev2_variance_12_read_readvariableop'
#savev2_count_12_read_readvariableop	&
"savev2_mean_13_read_readvariableop*
&savev2_variance_13_read_readvariableop'
#savev2_count_13_read_readvariableop	&
"savev2_mean_14_read_readvariableop*
&savev2_variance_14_read_readvariableop'
#savev2_count_14_read_readvariableop	&
"savev2_mean_15_read_readvariableop*
&savev2_variance_15_read_readvariableop'
#savev2_count_15_read_readvariableop	&
"savev2_mean_16_read_readvariableop*
&savev2_variance_16_read_readvariableop'
#savev2_count_16_read_readvariableop	&
"savev2_mean_17_read_readvariableop*
&savev2_variance_17_read_readvariableop'
#savev2_count_17_read_readvariableop	&
"savev2_mean_18_read_readvariableop*
&savev2_variance_18_read_readvariableop'
#savev2_count_18_read_readvariableop	&
"savev2_mean_19_read_readvariableop*
&savev2_variance_19_read_readvariableop'
#savev2_count_19_read_readvariableop	+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop$
 savev2_total_read_readvariableop'
#savev2_count_20_read_readvariableop&
"savev2_total_1_read_readvariableop'
#savev2_count_21_read_readvariableop
savev2_const_40

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:E*
dtype0*?
value?B?EB4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-1/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-2/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-3/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-4/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-5/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-6/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-7/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-8/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-9/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/count/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/mean/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-10/variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/count/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/mean/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-11/variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/count/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/mean/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-12/variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/count/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/mean/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-13/variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/count/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/mean/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-14/variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/count/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/mean/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-15/variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/count/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/mean/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-16/variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/count/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/mean/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-17/variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/count/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/mean/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-18/variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-18/count/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/mean/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-19/variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-19/count/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:E*
dtype0*?
value?B?EB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_mean_read_readvariableop#savev2_variance_read_readvariableop savev2_count_read_readvariableop!savev2_mean_1_read_readvariableop%savev2_variance_1_read_readvariableop"savev2_count_1_read_readvariableop!savev2_mean_2_read_readvariableop%savev2_variance_2_read_readvariableop"savev2_count_2_read_readvariableop!savev2_mean_3_read_readvariableop%savev2_variance_3_read_readvariableop"savev2_count_3_read_readvariableop!savev2_mean_4_read_readvariableop%savev2_variance_4_read_readvariableop"savev2_count_4_read_readvariableop!savev2_mean_5_read_readvariableop%savev2_variance_5_read_readvariableop"savev2_count_5_read_readvariableop!savev2_mean_6_read_readvariableop%savev2_variance_6_read_readvariableop"savev2_count_6_read_readvariableop!savev2_mean_7_read_readvariableop%savev2_variance_7_read_readvariableop"savev2_count_7_read_readvariableop!savev2_mean_8_read_readvariableop%savev2_variance_8_read_readvariableop"savev2_count_8_read_readvariableop!savev2_mean_9_read_readvariableop%savev2_variance_9_read_readvariableop"savev2_count_9_read_readvariableop"savev2_mean_10_read_readvariableop&savev2_variance_10_read_readvariableop#savev2_count_10_read_readvariableop"savev2_mean_11_read_readvariableop&savev2_variance_11_read_readvariableop#savev2_count_11_read_readvariableop"savev2_mean_12_read_readvariableop&savev2_variance_12_read_readvariableop#savev2_count_12_read_readvariableop"savev2_mean_13_read_readvariableop&savev2_variance_13_read_readvariableop#savev2_count_13_read_readvariableop"savev2_mean_14_read_readvariableop&savev2_variance_14_read_readvariableop#savev2_count_14_read_readvariableop"savev2_mean_15_read_readvariableop&savev2_variance_15_read_readvariableop#savev2_count_15_read_readvariableop"savev2_mean_16_read_readvariableop&savev2_variance_16_read_readvariableop#savev2_count_16_read_readvariableop"savev2_mean_17_read_readvariableop&savev2_variance_17_read_readvariableop#savev2_count_17_read_readvariableop"savev2_mean_18_read_readvariableop&savev2_variance_18_read_readvariableop#savev2_count_18_read_readvariableop"savev2_mean_19_read_readvariableop&savev2_variance_19_read_readvariableop#savev2_count_19_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop savev2_total_read_readvariableop#savev2_count_20_read_readvariableop"savev2_total_1_read_readvariableop#savev2_count_21_read_readvariableopsavev2_const_40"/device:CPU:0*
_output_shapes
 *S
dtypesI
G2E																				?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : :: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: :4

_output_shapes
: :5

_output_shapes
: :6

_output_shapes
: :7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: ::

_output_shapes
: :;

_output_shapes
: :<

_output_shapes
: :$= 

_output_shapes

: : >

_output_shapes
: :$? 

_output_shapes

: : @

_output_shapes
::A

_output_shapes
: :B

_output_shapes
: :C

_output_shapes
: :D

_output_shapes
: :E

_output_shapes
: 
?
?
'__inference_dense_1_layer_call_fn_10732

inputs
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_7957o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
??
?
?__inference_model_layer_call_and_return_conditional_losses_8984
battery_power
blue
clock_speed
dual_sim
fc

four_g

int_memory	
m_dep
	mobile_wt
n_cores
pc
	px_height
px_width
ram
sc_h
sc_w
	talk_time
three_g
touch_screen
wifi
normalization_sub_y
normalization_sqrt_x
normalization_1_sub_y
normalization_1_sqrt_x
normalization_2_sub_y
normalization_2_sqrt_x
normalization_3_sub_y
normalization_3_sqrt_x
normalization_4_sub_y
normalization_4_sqrt_x
normalization_5_sub_y
normalization_5_sqrt_x
normalization_6_sub_y
normalization_6_sqrt_x
normalization_7_sub_y
normalization_7_sqrt_x
normalization_8_sub_y
normalization_8_sqrt_x
normalization_9_sub_y
normalization_9_sqrt_x
normalization_10_sub_y
normalization_10_sqrt_x
normalization_11_sub_y
normalization_11_sqrt_x
normalization_12_sub_y
normalization_12_sqrt_x
normalization_13_sub_y
normalization_13_sqrt_x
normalization_14_sub_y
normalization_14_sqrt_x
normalization_15_sub_y
normalization_15_sqrt_x
normalization_16_sub_y
normalization_16_sqrt_x
normalization_17_sub_y
normalization_17_sqrt_x
normalization_18_sub_y
normalization_18_sqrt_x
normalization_19_sub_y
normalization_19_sqrt_x

dense_8972: 

dense_8974: 
dense_1_8978: 
dense_1_8980:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dropout/StatefulPartitionedCalln
normalization/subSubbattery_powernormalization_sub_y*
T0*'
_output_shapes
:?????????U
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes
:\
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????i
normalization_1/subSubbluenormalization_1_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_1/SqrtSqrtnormalization_1_sqrt_x*
T0*
_output_shapes
:^
normalization_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_1/MaximumMaximumnormalization_1/Sqrt:y:0"normalization_1/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_1/truedivRealDivnormalization_1/sub:z:0normalization_1/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_2/subSubclock_speednormalization_2_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_2/SqrtSqrtnormalization_2_sqrt_x*
T0*
_output_shapes
:^
normalization_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_2/MaximumMaximumnormalization_2/Sqrt:y:0"normalization_2/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_2/truedivRealDivnormalization_2/sub:z:0normalization_2/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_3/subSubdual_simnormalization_3_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_3/SqrtSqrtnormalization_3_sqrt_x*
T0*
_output_shapes
:^
normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*'
_output_shapes
:?????????g
normalization_4/subSubfcnormalization_4_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_4/SqrtSqrtnormalization_4_sqrt_x*
T0*
_output_shapes
:^
normalization_4/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_4/MaximumMaximumnormalization_4/Sqrt:y:0"normalization_4/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_4/truedivRealDivnormalization_4/sub:z:0normalization_4/Maximum:z:0*
T0*'
_output_shapes
:?????????k
normalization_5/subSubfour_gnormalization_5_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_5/SqrtSqrtnormalization_5_sqrt_x*
T0*
_output_shapes
:^
normalization_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_5/MaximumMaximumnormalization_5/Sqrt:y:0"normalization_5/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_5/truedivRealDivnormalization_5/sub:z:0normalization_5/Maximum:z:0*
T0*'
_output_shapes
:?????????o
normalization_6/subSub
int_memorynormalization_6_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_6/SqrtSqrtnormalization_6_sqrt_x*
T0*
_output_shapes
:^
normalization_6/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_6/MaximumMaximumnormalization_6/Sqrt:y:0"normalization_6/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_6/truedivRealDivnormalization_6/sub:z:0normalization_6/Maximum:z:0*
T0*'
_output_shapes
:?????????j
normalization_7/subSubm_depnormalization_7_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_7/SqrtSqrtnormalization_7_sqrt_x*
T0*
_output_shapes
:^
normalization_7/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_7/MaximumMaximumnormalization_7/Sqrt:y:0"normalization_7/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_7/truedivRealDivnormalization_7/sub:z:0normalization_7/Maximum:z:0*
T0*'
_output_shapes
:?????????n
normalization_8/subSub	mobile_wtnormalization_8_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_8/SqrtSqrtnormalization_8_sqrt_x*
T0*
_output_shapes
:^
normalization_8/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_8/MaximumMaximumnormalization_8/Sqrt:y:0"normalization_8/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_8/truedivRealDivnormalization_8/sub:z:0normalization_8/Maximum:z:0*
T0*'
_output_shapes
:?????????l
normalization_9/subSubn_coresnormalization_9_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_9/SqrtSqrtnormalization_9_sqrt_x*
T0*
_output_shapes
:^
normalization_9/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_9/MaximumMaximumnormalization_9/Sqrt:y:0"normalization_9/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_9/truedivRealDivnormalization_9/sub:z:0normalization_9/Maximum:z:0*
T0*'
_output_shapes
:?????????i
normalization_10/subSubpcnormalization_10_sub_y*
T0*'
_output_shapes
:?????????[
normalization_10/SqrtSqrtnormalization_10_sqrt_x*
T0*
_output_shapes
:_
normalization_10/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_10/MaximumMaximumnormalization_10/Sqrt:y:0#normalization_10/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_10/truedivRealDivnormalization_10/sub:z:0normalization_10/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_11/subSub	px_heightnormalization_11_sub_y*
T0*'
_output_shapes
:?????????[
normalization_11/SqrtSqrtnormalization_11_sqrt_x*
T0*
_output_shapes
:_
normalization_11/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_11/MaximumMaximumnormalization_11/Sqrt:y:0#normalization_11/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_11/truedivRealDivnormalization_11/sub:z:0normalization_11/Maximum:z:0*
T0*'
_output_shapes
:?????????o
normalization_12/subSubpx_widthnormalization_12_sub_y*
T0*'
_output_shapes
:?????????[
normalization_12/SqrtSqrtnormalization_12_sqrt_x*
T0*
_output_shapes
:_
normalization_12/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_12/MaximumMaximumnormalization_12/Sqrt:y:0#normalization_12/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_12/truedivRealDivnormalization_12/sub:z:0normalization_12/Maximum:z:0*
T0*'
_output_shapes
:?????????j
normalization_13/subSubramnormalization_13_sub_y*
T0*'
_output_shapes
:?????????[
normalization_13/SqrtSqrtnormalization_13_sqrt_x*
T0*
_output_shapes
:_
normalization_13/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_13/MaximumMaximumnormalization_13/Sqrt:y:0#normalization_13/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_13/truedivRealDivnormalization_13/sub:z:0normalization_13/Maximum:z:0*
T0*'
_output_shapes
:?????????k
normalization_14/subSubsc_hnormalization_14_sub_y*
T0*'
_output_shapes
:?????????[
normalization_14/SqrtSqrtnormalization_14_sqrt_x*
T0*
_output_shapes
:_
normalization_14/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_14/MaximumMaximumnormalization_14/Sqrt:y:0#normalization_14/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_14/truedivRealDivnormalization_14/sub:z:0normalization_14/Maximum:z:0*
T0*'
_output_shapes
:?????????k
normalization_15/subSubsc_wnormalization_15_sub_y*
T0*'
_output_shapes
:?????????[
normalization_15/SqrtSqrtnormalization_15_sqrt_x*
T0*
_output_shapes
:_
normalization_15/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_15/MaximumMaximumnormalization_15/Sqrt:y:0#normalization_15/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_15/truedivRealDivnormalization_15/sub:z:0normalization_15/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_16/subSub	talk_timenormalization_16_sub_y*
T0*'
_output_shapes
:?????????[
normalization_16/SqrtSqrtnormalization_16_sqrt_x*
T0*
_output_shapes
:_
normalization_16/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_16/MaximumMaximumnormalization_16/Sqrt:y:0#normalization_16/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_16/truedivRealDivnormalization_16/sub:z:0normalization_16/Maximum:z:0*
T0*'
_output_shapes
:?????????n
normalization_17/subSubthree_gnormalization_17_sub_y*
T0*'
_output_shapes
:?????????[
normalization_17/SqrtSqrtnormalization_17_sqrt_x*
T0*
_output_shapes
:_
normalization_17/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_17/MaximumMaximumnormalization_17/Sqrt:y:0#normalization_17/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_17/truedivRealDivnormalization_17/sub:z:0normalization_17/Maximum:z:0*
T0*'
_output_shapes
:?????????s
normalization_18/subSubtouch_screennormalization_18_sub_y*
T0*'
_output_shapes
:?????????[
normalization_18/SqrtSqrtnormalization_18_sqrt_x*
T0*
_output_shapes
:_
normalization_18/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_18/MaximumMaximumnormalization_18/Sqrt:y:0#normalization_18/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_18/truedivRealDivnormalization_18/sub:z:0normalization_18/Maximum:z:0*
T0*'
_output_shapes
:?????????k
normalization_19/subSubwifinormalization_19_sub_y*
T0*'
_output_shapes
:?????????[
normalization_19/SqrtSqrtnormalization_19_sqrt_x*
T0*
_output_shapes
:_
normalization_19/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_19/MaximumMaximumnormalization_19/Sqrt:y:0#normalization_19/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_19/truedivRealDivnormalization_19/sub:z:0normalization_19/Maximum:z:0*
T0*'
_output_shapes
:??????????
concatenate/PartitionedCallPartitionedCallnormalization/truediv:z:0normalization_1/truediv:z:0normalization_2/truediv:z:0normalization_3/truediv:z:0normalization_4/truediv:z:0normalization_5/truediv:z:0normalization_6/truediv:z:0normalization_7/truediv:z:0normalization_8/truediv:z:0normalization_9/truediv:z:0normalization_10/truediv:z:0normalization_11/truediv:z:0normalization_12/truediv:z:0normalization_13/truediv:z:0normalization_14/truediv:z:0normalization_15/truediv:z:0normalization_16/truediv:z:0normalization_17/truediv:z:0normalization_18/truediv:z:0normalization_19/truediv:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_7920?
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0
dense_8972
dense_8974*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_7933?
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_8085?
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_8978dense_1_8980*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_7957w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????::::::::::::::::::::::::::::::::::::::::: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namebattery_power:MI
'
_output_shapes
:?????????

_user_specified_nameblue:TP
'
_output_shapes
:?????????
%
_user_specified_nameclock_speed:QM
'
_output_shapes
:?????????
"
_user_specified_name
dual_sim:KG
'
_output_shapes
:?????????

_user_specified_namefc:OK
'
_output_shapes
:?????????
 
_user_specified_namefour_g:SO
'
_output_shapes
:?????????
$
_user_specified_name
int_memory:NJ
'
_output_shapes
:?????????

_user_specified_namem_dep:RN
'
_output_shapes
:?????????
#
_user_specified_name	mobile_wt:P	L
'
_output_shapes
:?????????
!
_user_specified_name	n_cores:K
G
'
_output_shapes
:?????????

_user_specified_namepc:RN
'
_output_shapes
:?????????
#
_user_specified_name	px_height:QM
'
_output_shapes
:?????????
"
_user_specified_name
px_width:LH
'
_output_shapes
:?????????

_user_specified_nameram:MI
'
_output_shapes
:?????????

_user_specified_namesc_h:MI
'
_output_shapes
:?????????

_user_specified_namesc_w:RN
'
_output_shapes
:?????????
#
_user_specified_name	talk_time:PL
'
_output_shapes
:?????????
!
_user_specified_name	three_g:UQ
'
_output_shapes
:?????????
&
_user_specified_nametouch_screen:MI
'
_output_shapes
:?????????

_user_specified_namewifi: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::  

_output_shapes
:: !

_output_shapes
:: "

_output_shapes
:: #

_output_shapes
:: $

_output_shapes
:: %

_output_shapes
:: &

_output_shapes
:: '

_output_shapes
:: (

_output_shapes
:: )

_output_shapes
:: *

_output_shapes
:: +

_output_shapes
:: ,

_output_shapes
:: -

_output_shapes
:: .

_output_shapes
:: /

_output_shapes
:: 0

_output_shapes
:: 1

_output_shapes
:: 2

_output_shapes
:: 3

_output_shapes
:: 4

_output_shapes
:: 5

_output_shapes
:: 6

_output_shapes
:: 7

_output_shapes
:: 8

_output_shapes
:: 9

_output_shapes
:: :

_output_shapes
:: ;

_output_shapes
:
?&
?
__inference_adapt_step_10063
iterator

iterator_1%
add_readvariableop_resource:	 !
readvariableop_resource: #
readvariableop_2_resource: ??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:?????????*"
output_shapes
:?????????*
output_types
2h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/meanMeanIteratorGetNext:components:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(`
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:?
moments/SquaredDifferenceSquaredDifferenceIteratorGetNext:components:0moments/StopGradient:output:0*
T0*#
_output_shapes
:?????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(i
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 o
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 a
ShapeShapeIteratorGetNext:components:0*
T0*
_output_shapes
:*
out_type0	Z
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:O
ConstConst*
_output_shapes
:*
dtype0*
valueB: P
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: f
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	X
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: K
CastCastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: G
Cast_1Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: I
truedivRealDivCast:y:0
Cast_1:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: ^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
: T
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
: C
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
: `
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0R
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
: J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @F
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
: b
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype0R
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
: A
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
: R
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
: L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
: V
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
: E
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
: E
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: ?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype0?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
?
C
'__inference_dropout_layer_call_fn_10701

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_7944`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?'
?
__inference_adapt_step_9783
iterator

iterator_1%
add_readvariableop_resource:	 !
readvariableop_resource: #
readvariableop_2_resource: ??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:?????????*"
output_shapes
:?????????*
output_types
2g
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*#
_output_shapes
:?????????h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: }
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(`
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:?
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*#
_output_shapes
:?????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(i
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 o
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 M
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	Z
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:O
ConstConst*
_output_shapes
:*
dtype0*
valueB: P
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: f
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	X
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: M
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: G
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: K
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: ^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
: T
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
: C
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
: `
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0R
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
: J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @F
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
: b
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype0R
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
: A
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
: R
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
: L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
: V
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
: E
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
: E
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: ?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype0?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
??
?
__inference__wrapped_model_7710
battery_power
blue
clock_speed
dual_sim
fc

four_g

int_memory	
m_dep
	mobile_wt
n_cores
pc
	px_height
px_width
ram
sc_h
sc_w
	talk_time
three_g
touch_screen
wifi
model_normalization_sub_y
model_normalization_sqrt_x
model_normalization_1_sub_y 
model_normalization_1_sqrt_x
model_normalization_2_sub_y 
model_normalization_2_sqrt_x
model_normalization_3_sub_y 
model_normalization_3_sqrt_x
model_normalization_4_sub_y 
model_normalization_4_sqrt_x
model_normalization_5_sub_y 
model_normalization_5_sqrt_x
model_normalization_6_sub_y 
model_normalization_6_sqrt_x
model_normalization_7_sub_y 
model_normalization_7_sqrt_x
model_normalization_8_sub_y 
model_normalization_8_sqrt_x
model_normalization_9_sub_y 
model_normalization_9_sqrt_x 
model_normalization_10_sub_y!
model_normalization_10_sqrt_x 
model_normalization_11_sub_y!
model_normalization_11_sqrt_x 
model_normalization_12_sub_y!
model_normalization_12_sqrt_x 
model_normalization_13_sub_y!
model_normalization_13_sqrt_x 
model_normalization_14_sub_y!
model_normalization_14_sqrt_x 
model_normalization_15_sub_y!
model_normalization_15_sqrt_x 
model_normalization_16_sub_y!
model_normalization_16_sqrt_x 
model_normalization_17_sub_y!
model_normalization_17_sqrt_x 
model_normalization_18_sub_y!
model_normalization_18_sqrt_x 
model_normalization_19_sub_y!
model_normalization_19_sqrt_x<
*model_dense_matmul_readvariableop_resource: 9
+model_dense_biasadd_readvariableop_resource: >
,model_dense_1_matmul_readvariableop_resource: ;
-model_dense_1_biasadd_readvariableop_resource:
identity??"model/dense/BiasAdd/ReadVariableOp?!model/dense/MatMul/ReadVariableOp?$model/dense_1/BiasAdd/ReadVariableOp?#model/dense_1/MatMul/ReadVariableOpz
model/normalization/subSubbattery_powermodel_normalization_sub_y*
T0*'
_output_shapes
:?????????a
model/normalization/SqrtSqrtmodel_normalization_sqrt_x*
T0*
_output_shapes
:b
model/normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
model/normalization/MaximumMaximummodel/normalization/Sqrt:y:0&model/normalization/Maximum/y:output:0*
T0*
_output_shapes
:?
model/normalization/truedivRealDivmodel/normalization/sub:z:0model/normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????u
model/normalization_1/subSubbluemodel_normalization_1_sub_y*
T0*'
_output_shapes
:?????????e
model/normalization_1/SqrtSqrtmodel_normalization_1_sqrt_x*
T0*
_output_shapes
:d
model/normalization_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
model/normalization_1/MaximumMaximummodel/normalization_1/Sqrt:y:0(model/normalization_1/Maximum/y:output:0*
T0*
_output_shapes
:?
model/normalization_1/truedivRealDivmodel/normalization_1/sub:z:0!model/normalization_1/Maximum:z:0*
T0*'
_output_shapes
:?????????|
model/normalization_2/subSubclock_speedmodel_normalization_2_sub_y*
T0*'
_output_shapes
:?????????e
model/normalization_2/SqrtSqrtmodel_normalization_2_sqrt_x*
T0*
_output_shapes
:d
model/normalization_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
model/normalization_2/MaximumMaximummodel/normalization_2/Sqrt:y:0(model/normalization_2/Maximum/y:output:0*
T0*
_output_shapes
:?
model/normalization_2/truedivRealDivmodel/normalization_2/sub:z:0!model/normalization_2/Maximum:z:0*
T0*'
_output_shapes
:?????????y
model/normalization_3/subSubdual_simmodel_normalization_3_sub_y*
T0*'
_output_shapes
:?????????e
model/normalization_3/SqrtSqrtmodel_normalization_3_sqrt_x*
T0*
_output_shapes
:d
model/normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
model/normalization_3/MaximumMaximummodel/normalization_3/Sqrt:y:0(model/normalization_3/Maximum/y:output:0*
T0*
_output_shapes
:?
model/normalization_3/truedivRealDivmodel/normalization_3/sub:z:0!model/normalization_3/Maximum:z:0*
T0*'
_output_shapes
:?????????s
model/normalization_4/subSubfcmodel_normalization_4_sub_y*
T0*'
_output_shapes
:?????????e
model/normalization_4/SqrtSqrtmodel_normalization_4_sqrt_x*
T0*
_output_shapes
:d
model/normalization_4/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
model/normalization_4/MaximumMaximummodel/normalization_4/Sqrt:y:0(model/normalization_4/Maximum/y:output:0*
T0*
_output_shapes
:?
model/normalization_4/truedivRealDivmodel/normalization_4/sub:z:0!model/normalization_4/Maximum:z:0*
T0*'
_output_shapes
:?????????w
model/normalization_5/subSubfour_gmodel_normalization_5_sub_y*
T0*'
_output_shapes
:?????????e
model/normalization_5/SqrtSqrtmodel_normalization_5_sqrt_x*
T0*
_output_shapes
:d
model/normalization_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
model/normalization_5/MaximumMaximummodel/normalization_5/Sqrt:y:0(model/normalization_5/Maximum/y:output:0*
T0*
_output_shapes
:?
model/normalization_5/truedivRealDivmodel/normalization_5/sub:z:0!model/normalization_5/Maximum:z:0*
T0*'
_output_shapes
:?????????{
model/normalization_6/subSub
int_memorymodel_normalization_6_sub_y*
T0*'
_output_shapes
:?????????e
model/normalization_6/SqrtSqrtmodel_normalization_6_sqrt_x*
T0*
_output_shapes
:d
model/normalization_6/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
model/normalization_6/MaximumMaximummodel/normalization_6/Sqrt:y:0(model/normalization_6/Maximum/y:output:0*
T0*
_output_shapes
:?
model/normalization_6/truedivRealDivmodel/normalization_6/sub:z:0!model/normalization_6/Maximum:z:0*
T0*'
_output_shapes
:?????????v
model/normalization_7/subSubm_depmodel_normalization_7_sub_y*
T0*'
_output_shapes
:?????????e
model/normalization_7/SqrtSqrtmodel_normalization_7_sqrt_x*
T0*
_output_shapes
:d
model/normalization_7/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
model/normalization_7/MaximumMaximummodel/normalization_7/Sqrt:y:0(model/normalization_7/Maximum/y:output:0*
T0*
_output_shapes
:?
model/normalization_7/truedivRealDivmodel/normalization_7/sub:z:0!model/normalization_7/Maximum:z:0*
T0*'
_output_shapes
:?????????z
model/normalization_8/subSub	mobile_wtmodel_normalization_8_sub_y*
T0*'
_output_shapes
:?????????e
model/normalization_8/SqrtSqrtmodel_normalization_8_sqrt_x*
T0*
_output_shapes
:d
model/normalization_8/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
model/normalization_8/MaximumMaximummodel/normalization_8/Sqrt:y:0(model/normalization_8/Maximum/y:output:0*
T0*
_output_shapes
:?
model/normalization_8/truedivRealDivmodel/normalization_8/sub:z:0!model/normalization_8/Maximum:z:0*
T0*'
_output_shapes
:?????????x
model/normalization_9/subSubn_coresmodel_normalization_9_sub_y*
T0*'
_output_shapes
:?????????e
model/normalization_9/SqrtSqrtmodel_normalization_9_sqrt_x*
T0*
_output_shapes
:d
model/normalization_9/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
model/normalization_9/MaximumMaximummodel/normalization_9/Sqrt:y:0(model/normalization_9/Maximum/y:output:0*
T0*
_output_shapes
:?
model/normalization_9/truedivRealDivmodel/normalization_9/sub:z:0!model/normalization_9/Maximum:z:0*
T0*'
_output_shapes
:?????????u
model/normalization_10/subSubpcmodel_normalization_10_sub_y*
T0*'
_output_shapes
:?????????g
model/normalization_10/SqrtSqrtmodel_normalization_10_sqrt_x*
T0*
_output_shapes
:e
 model/normalization_10/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
model/normalization_10/MaximumMaximummodel/normalization_10/Sqrt:y:0)model/normalization_10/Maximum/y:output:0*
T0*
_output_shapes
:?
model/normalization_10/truedivRealDivmodel/normalization_10/sub:z:0"model/normalization_10/Maximum:z:0*
T0*'
_output_shapes
:?????????|
model/normalization_11/subSub	px_heightmodel_normalization_11_sub_y*
T0*'
_output_shapes
:?????????g
model/normalization_11/SqrtSqrtmodel_normalization_11_sqrt_x*
T0*
_output_shapes
:e
 model/normalization_11/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
model/normalization_11/MaximumMaximummodel/normalization_11/Sqrt:y:0)model/normalization_11/Maximum/y:output:0*
T0*
_output_shapes
:?
model/normalization_11/truedivRealDivmodel/normalization_11/sub:z:0"model/normalization_11/Maximum:z:0*
T0*'
_output_shapes
:?????????{
model/normalization_12/subSubpx_widthmodel_normalization_12_sub_y*
T0*'
_output_shapes
:?????????g
model/normalization_12/SqrtSqrtmodel_normalization_12_sqrt_x*
T0*
_output_shapes
:e
 model/normalization_12/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
model/normalization_12/MaximumMaximummodel/normalization_12/Sqrt:y:0)model/normalization_12/Maximum/y:output:0*
T0*
_output_shapes
:?
model/normalization_12/truedivRealDivmodel/normalization_12/sub:z:0"model/normalization_12/Maximum:z:0*
T0*'
_output_shapes
:?????????v
model/normalization_13/subSubrammodel_normalization_13_sub_y*
T0*'
_output_shapes
:?????????g
model/normalization_13/SqrtSqrtmodel_normalization_13_sqrt_x*
T0*
_output_shapes
:e
 model/normalization_13/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
model/normalization_13/MaximumMaximummodel/normalization_13/Sqrt:y:0)model/normalization_13/Maximum/y:output:0*
T0*
_output_shapes
:?
model/normalization_13/truedivRealDivmodel/normalization_13/sub:z:0"model/normalization_13/Maximum:z:0*
T0*'
_output_shapes
:?????????w
model/normalization_14/subSubsc_hmodel_normalization_14_sub_y*
T0*'
_output_shapes
:?????????g
model/normalization_14/SqrtSqrtmodel_normalization_14_sqrt_x*
T0*
_output_shapes
:e
 model/normalization_14/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
model/normalization_14/MaximumMaximummodel/normalization_14/Sqrt:y:0)model/normalization_14/Maximum/y:output:0*
T0*
_output_shapes
:?
model/normalization_14/truedivRealDivmodel/normalization_14/sub:z:0"model/normalization_14/Maximum:z:0*
T0*'
_output_shapes
:?????????w
model/normalization_15/subSubsc_wmodel_normalization_15_sub_y*
T0*'
_output_shapes
:?????????g
model/normalization_15/SqrtSqrtmodel_normalization_15_sqrt_x*
T0*
_output_shapes
:e
 model/normalization_15/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
model/normalization_15/MaximumMaximummodel/normalization_15/Sqrt:y:0)model/normalization_15/Maximum/y:output:0*
T0*
_output_shapes
:?
model/normalization_15/truedivRealDivmodel/normalization_15/sub:z:0"model/normalization_15/Maximum:z:0*
T0*'
_output_shapes
:?????????|
model/normalization_16/subSub	talk_timemodel_normalization_16_sub_y*
T0*'
_output_shapes
:?????????g
model/normalization_16/SqrtSqrtmodel_normalization_16_sqrt_x*
T0*
_output_shapes
:e
 model/normalization_16/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
model/normalization_16/MaximumMaximummodel/normalization_16/Sqrt:y:0)model/normalization_16/Maximum/y:output:0*
T0*
_output_shapes
:?
model/normalization_16/truedivRealDivmodel/normalization_16/sub:z:0"model/normalization_16/Maximum:z:0*
T0*'
_output_shapes
:?????????z
model/normalization_17/subSubthree_gmodel_normalization_17_sub_y*
T0*'
_output_shapes
:?????????g
model/normalization_17/SqrtSqrtmodel_normalization_17_sqrt_x*
T0*
_output_shapes
:e
 model/normalization_17/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
model/normalization_17/MaximumMaximummodel/normalization_17/Sqrt:y:0)model/normalization_17/Maximum/y:output:0*
T0*
_output_shapes
:?
model/normalization_17/truedivRealDivmodel/normalization_17/sub:z:0"model/normalization_17/Maximum:z:0*
T0*'
_output_shapes
:?????????
model/normalization_18/subSubtouch_screenmodel_normalization_18_sub_y*
T0*'
_output_shapes
:?????????g
model/normalization_18/SqrtSqrtmodel_normalization_18_sqrt_x*
T0*
_output_shapes
:e
 model/normalization_18/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
model/normalization_18/MaximumMaximummodel/normalization_18/Sqrt:y:0)model/normalization_18/Maximum/y:output:0*
T0*
_output_shapes
:?
model/normalization_18/truedivRealDivmodel/normalization_18/sub:z:0"model/normalization_18/Maximum:z:0*
T0*'
_output_shapes
:?????????w
model/normalization_19/subSubwifimodel_normalization_19_sub_y*
T0*'
_output_shapes
:?????????g
model/normalization_19/SqrtSqrtmodel_normalization_19_sqrt_x*
T0*
_output_shapes
:e
 model/normalization_19/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
model/normalization_19/MaximumMaximummodel/normalization_19/Sqrt:y:0)model/normalization_19/Maximum/y:output:0*
T0*
_output_shapes
:?
model/normalization_19/truedivRealDivmodel/normalization_19/sub:z:0"model/normalization_19/Maximum:z:0*
T0*'
_output_shapes
:?????????_
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
model/concatenate/concatConcatV2model/normalization/truediv:z:0!model/normalization_1/truediv:z:0!model/normalization_2/truediv:z:0!model/normalization_3/truediv:z:0!model/normalization_4/truediv:z:0!model/normalization_5/truediv:z:0!model/normalization_6/truediv:z:0!model/normalization_7/truediv:z:0!model/normalization_8/truediv:z:0!model/normalization_9/truediv:z:0"model/normalization_10/truediv:z:0"model/normalization_11/truediv:z:0"model/normalization_12/truediv:z:0"model/normalization_13/truediv:z:0"model/normalization_14/truediv:z:0"model/normalization_15/truediv:z:0"model/normalization_16/truediv:z:0"model/normalization_17/truediv:z:0"model/normalization_18/truediv:z:0"model/normalization_19/truediv:z:0&model/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:??????????
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
model/dense/MatMulMatMul!model/concatenate/concat:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? h
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:????????? t
model/dropout/IdentityIdentitymodel/dense/Relu:activations:0*
T0*'
_output_shapes
:????????? ?
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
model/dense_1/MatMulMatMulmodel/dropout/Identity:output:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
model/dense_1/SigmoidSigmoidmodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitymodel/dense_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????::::::::::::::::::::::::::::::::::::::::: : : : 2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp:V R
'
_output_shapes
:?????????
'
_user_specified_namebattery_power:MI
'
_output_shapes
:?????????

_user_specified_nameblue:TP
'
_output_shapes
:?????????
%
_user_specified_nameclock_speed:QM
'
_output_shapes
:?????????
"
_user_specified_name
dual_sim:KG
'
_output_shapes
:?????????

_user_specified_namefc:OK
'
_output_shapes
:?????????
 
_user_specified_namefour_g:SO
'
_output_shapes
:?????????
$
_user_specified_name
int_memory:NJ
'
_output_shapes
:?????????

_user_specified_namem_dep:RN
'
_output_shapes
:?????????
#
_user_specified_name	mobile_wt:P	L
'
_output_shapes
:?????????
!
_user_specified_name	n_cores:K
G
'
_output_shapes
:?????????

_user_specified_namepc:RN
'
_output_shapes
:?????????
#
_user_specified_name	px_height:QM
'
_output_shapes
:?????????
"
_user_specified_name
px_width:LH
'
_output_shapes
:?????????

_user_specified_nameram:MI
'
_output_shapes
:?????????

_user_specified_namesc_h:MI
'
_output_shapes
:?????????

_user_specified_namesc_w:RN
'
_output_shapes
:?????????
#
_user_specified_name	talk_time:PL
'
_output_shapes
:?????????
!
_user_specified_name	three_g:UQ
'
_output_shapes
:?????????
&
_user_specified_nametouch_screen:MI
'
_output_shapes
:?????????

_user_specified_namewifi: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::  

_output_shapes
:: !

_output_shapes
:: "

_output_shapes
:: #

_output_shapes
:: $

_output_shapes
:: %

_output_shapes
:: &

_output_shapes
:: '

_output_shapes
:: (

_output_shapes
:: )

_output_shapes
:: *

_output_shapes
:: +

_output_shapes
:: ,

_output_shapes
:: -

_output_shapes
:: .

_output_shapes
:: /

_output_shapes
:: 0

_output_shapes
:: 1

_output_shapes
:: 2

_output_shapes
:: 3

_output_shapes
:: 4

_output_shapes
:: 5

_output_shapes
:: 6

_output_shapes
:: 7

_output_shapes
:: 8

_output_shapes
:: 9

_output_shapes
:: :

_output_shapes
:: ;

_output_shapes
:
?'
?
__inference_adapt_step_9970
iterator

iterator_1%
add_readvariableop_resource:	 !
readvariableop_resource: #
readvariableop_2_resource: ??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:?????????*"
output_shapes
:?????????*
output_types
2g
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*#
_output_shapes
:?????????h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: }
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(`
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:?
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*#
_output_shapes
:?????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(i
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 o
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 M
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	Z
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:O
ConstConst*
_output_shapes
:*
dtype0*
valueB: P
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: f
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	X
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: M
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: G
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: K
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: ^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
: T
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
: C
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
: `
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0R
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
: J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @F
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
: b
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype0R
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
: A
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
: R
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
: L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
: V
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
: E
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
: E
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: ?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype0?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
?'
?
__inference_adapt_step_10110
iterator

iterator_1%
add_readvariableop_resource:	 !
readvariableop_resource: #
readvariableop_2_resource: ??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:?????????*"
output_shapes
:?????????*
output_types
2g
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*#
_output_shapes
:?????????h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: }
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(`
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:?
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*#
_output_shapes
:?????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(i
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 o
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 M
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	Z
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:O
ConstConst*
_output_shapes
:*
dtype0*
valueB: P
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: f
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	X
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: M
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: G
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: K
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: ^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
: T
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
: C
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
: `
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0R
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
: J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @F
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
: b
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype0R
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
: A
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
: R
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
: L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
: V
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
: E
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
: E
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: ?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype0?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
?'
?
__inference_adapt_step_10204
iterator

iterator_1%
add_readvariableop_resource:	 !
readvariableop_resource: #
readvariableop_2_resource: ??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:?????????*"
output_shapes
:?????????*
output_types
2g
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*#
_output_shapes
:?????????h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: }
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(`
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:?
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*#
_output_shapes
:?????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(i
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 o
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 M
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	Z
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:O
ConstConst*
_output_shapes
:*
dtype0*
valueB: P
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: f
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	X
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: M
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: G
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: K
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: ^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
: T
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
: C
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
: `
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0R
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
: J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @F
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
: b
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype0R
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
: A
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
: R
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
: L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
: V
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
: E
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
: E
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: ?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype0?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
?'
?
__inference_adapt_step_10251
iterator

iterator_1%
add_readvariableop_resource:	 !
readvariableop_resource: #
readvariableop_2_resource: ??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:?????????*"
output_shapes
:?????????*
output_types
2g
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*#
_output_shapes
:?????????h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: }
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(`
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:?
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*#
_output_shapes
:?????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(i
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 o
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 M
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	Z
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:O
ConstConst*
_output_shapes
:*
dtype0*
valueB: P
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: f
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	X
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: M
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: G
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: K
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: ^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
: T
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
: C
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
: `
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0R
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
: J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @F
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
: b
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype0R
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
: A
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
: R
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
: L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
: V
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
: E
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
: E
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: ?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype0?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
?
?
E__inference_concatenate_layer_call_and_return_conditional_losses_7920

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
	inputs_13
	inputs_14
	inputs_15
	inputs_16
	inputs_17
	inputs_18
	inputs_19
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12	inputs_13	inputs_14	inputs_15	inputs_16	inputs_17	inputs_18	inputs_19concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O	K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O
K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
F__inference_concatenate_layer_call_and_return_conditional_losses_10676
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
	inputs_13
	inputs_14
	inputs_15
	inputs_16
	inputs_17
	inputs_18
	inputs_19
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12	inputs_13	inputs_14	inputs_15	inputs_16	inputs_17	inputs_18	inputs_19concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/8:Q	M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/9:R
N
'
_output_shapes
:?????????
#
_user_specified_name	inputs/10:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/11:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/12:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/13:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/14:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/15:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/16:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/17:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/18:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/19
?

?
@__inference_dense_layer_call_and_return_conditional_losses_10696

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
B__inference_dense_1_layer_call_and_return_conditional_losses_10743

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?/
?
$__inference_model_layer_call_fn_8634
battery_power
blue
clock_speed
dual_sim
fc

four_g

int_memory	
m_dep
	mobile_wt
n_cores
pc
	px_height
px_width
ram
sc_h
sc_w
	talk_time
three_g
touch_screen
wifi
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39: 

unknown_40: 

unknown_41: 

unknown_42:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallbattery_powerblueclock_speeddual_simfcfour_g
int_memorym_dep	mobile_wtn_corespc	px_heightpx_widthramsc_hsc_w	talk_timethree_gtouch_screenwifiunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*K
TinD
B2@*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
<=>?*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_8431o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????::::::::::::::::::::::::::::::::::::::::: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namebattery_power:MI
'
_output_shapes
:?????????

_user_specified_nameblue:TP
'
_output_shapes
:?????????
%
_user_specified_nameclock_speed:QM
'
_output_shapes
:?????????
"
_user_specified_name
dual_sim:KG
'
_output_shapes
:?????????

_user_specified_namefc:OK
'
_output_shapes
:?????????
 
_user_specified_namefour_g:SO
'
_output_shapes
:?????????
$
_user_specified_name
int_memory:NJ
'
_output_shapes
:?????????

_user_specified_namem_dep:RN
'
_output_shapes
:?????????
#
_user_specified_name	mobile_wt:P	L
'
_output_shapes
:?????????
!
_user_specified_name	n_cores:K
G
'
_output_shapes
:?????????

_user_specified_namepc:RN
'
_output_shapes
:?????????
#
_user_specified_name	px_height:QM
'
_output_shapes
:?????????
"
_user_specified_name
px_width:LH
'
_output_shapes
:?????????

_user_specified_nameram:MI
'
_output_shapes
:?????????

_user_specified_namesc_h:MI
'
_output_shapes
:?????????

_user_specified_namesc_w:RN
'
_output_shapes
:?????????
#
_user_specified_name	talk_time:PL
'
_output_shapes
:?????????
!
_user_specified_name	three_g:UQ
'
_output_shapes
:?????????
&
_user_specified_nametouch_screen:MI
'
_output_shapes
:?????????

_user_specified_namewifi: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::  

_output_shapes
:: !

_output_shapes
:: "

_output_shapes
:: #

_output_shapes
:: $

_output_shapes
:: %

_output_shapes
:: &

_output_shapes
:: '

_output_shapes
:: (

_output_shapes
:: )

_output_shapes
:: *

_output_shapes
:: +

_output_shapes
:: ,

_output_shapes
:: -

_output_shapes
:: .

_output_shapes
:: /

_output_shapes
:: 0

_output_shapes
:: 1

_output_shapes
:: 2

_output_shapes
:: 3

_output_shapes
:: 4

_output_shapes
:: 5

_output_shapes
:: 6

_output_shapes
:: 7

_output_shapes
:: 8

_output_shapes
:: 9

_output_shapes
:: :

_output_shapes
:: ;

_output_shapes
:
?'
?
__inference_adapt_step_10580
iterator

iterator_1%
add_readvariableop_resource:	 !
readvariableop_resource: #
readvariableop_2_resource: ??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:?????????*"
output_shapes
:?????????*
output_types
2g
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*#
_output_shapes
:?????????h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: }
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(`
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:?
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*#
_output_shapes
:?????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(i
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 o
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 M
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	Z
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:O
ConstConst*
_output_shapes
:*
dtype0*
valueB: P
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: f
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	X
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: M
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: G
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: K
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: ^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
: T
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
: C
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
: `
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0R
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
: J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @F
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
: b
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype0R
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
: A
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
: R
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
: L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
: V
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
: E
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
: E
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: ?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype0?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
?
`
'__inference_dropout_layer_call_fn_10706

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_8085o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
a
B__inference_dropout_layer_call_and_return_conditional_losses_10723

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:????????? C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:????????? Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
?__inference_dense_layer_call_and_return_conditional_losses_7933

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?'
?
__inference_adapt_step_10533
iterator

iterator_1%
add_readvariableop_resource:	 !
readvariableop_resource: #
readvariableop_2_resource: ??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:?????????*"
output_shapes
:?????????*
output_types
2g
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*#
_output_shapes
:?????????h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: }
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(`
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:?
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*#
_output_shapes
:?????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(i
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 o
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 M
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	Z
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:O
ConstConst*
_output_shapes
:*
dtype0*
valueB: P
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: f
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	X
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: M
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: G
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: K
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: ^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
: T
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
: C
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
: `
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0R
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
: J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @F
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
: b
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype0R
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
: A
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
: R
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
: L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
: V
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
: E
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
: E
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: ?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype0?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
?/
?
$__inference_model_layer_call_fn_8055
battery_power
blue
clock_speed
dual_sim
fc

four_g

int_memory	
m_dep
	mobile_wt
n_cores
pc
	px_height
px_width
ram
sc_h
sc_w
	talk_time
three_g
touch_screen
wifi
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39: 

unknown_40: 

unknown_41: 

unknown_42:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallbattery_powerblueclock_speeddual_simfcfour_g
int_memorym_dep	mobile_wtn_corespc	px_heightpx_widthramsc_hsc_w	talk_timethree_gtouch_screenwifiunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*K
TinD
B2@*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
<=>?*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_7964o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????::::::::::::::::::::::::::::::::::::::::: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namebattery_power:MI
'
_output_shapes
:?????????

_user_specified_nameblue:TP
'
_output_shapes
:?????????
%
_user_specified_nameclock_speed:QM
'
_output_shapes
:?????????
"
_user_specified_name
dual_sim:KG
'
_output_shapes
:?????????

_user_specified_namefc:OK
'
_output_shapes
:?????????
 
_user_specified_namefour_g:SO
'
_output_shapes
:?????????
$
_user_specified_name
int_memory:NJ
'
_output_shapes
:?????????

_user_specified_namem_dep:RN
'
_output_shapes
:?????????
#
_user_specified_name	mobile_wt:P	L
'
_output_shapes
:?????????
!
_user_specified_name	n_cores:K
G
'
_output_shapes
:?????????

_user_specified_namepc:RN
'
_output_shapes
:?????????
#
_user_specified_name	px_height:QM
'
_output_shapes
:?????????
"
_user_specified_name
px_width:LH
'
_output_shapes
:?????????

_user_specified_nameram:MI
'
_output_shapes
:?????????

_user_specified_namesc_h:MI
'
_output_shapes
:?????????

_user_specified_namesc_w:RN
'
_output_shapes
:?????????
#
_user_specified_name	talk_time:PL
'
_output_shapes
:?????????
!
_user_specified_name	three_g:UQ
'
_output_shapes
:?????????
&
_user_specified_nametouch_screen:MI
'
_output_shapes
:?????????

_user_specified_namewifi: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::  

_output_shapes
:: !

_output_shapes
:: "

_output_shapes
:: #

_output_shapes
:: $

_output_shapes
:: %

_output_shapes
:: &

_output_shapes
:: '

_output_shapes
:: (

_output_shapes
:: )

_output_shapes
:: *

_output_shapes
:: +

_output_shapes
:: ,

_output_shapes
:: -

_output_shapes
:: .

_output_shapes
:: /

_output_shapes
:: 0

_output_shapes
:: 1

_output_shapes
:: 2

_output_shapes
:: 3

_output_shapes
:: 4

_output_shapes
:: 5

_output_shapes
:: 6

_output_shapes
:: 7

_output_shapes
:: 8

_output_shapes
:: 9

_output_shapes
:: :

_output_shapes
:: ;

_output_shapes
:
?'
?
__inference_adapt_step_10486
iterator

iterator_1%
add_readvariableop_resource:	 !
readvariableop_resource: #
readvariableop_2_resource: ??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:?????????*"
output_shapes
:?????????*
output_types
2g
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*#
_output_shapes
:?????????h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: }
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(`
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:?
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*#
_output_shapes
:?????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(i
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 o
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 M
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	Z
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:O
ConstConst*
_output_shapes
:*
dtype0*
valueB: P
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: f
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	X
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: M
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: G
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: K
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: ^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
: T
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
: C
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
: `
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0R
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
: J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @F
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
: b
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype0R
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
: A
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
: R
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
: L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
: V
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
: E
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
: E
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: ?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype0?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
?'
?
__inference_adapt_step_9876
iterator

iterator_1%
add_readvariableop_resource:	 !
readvariableop_resource: #
readvariableop_2_resource: ??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:?????????*"
output_shapes
:?????????*
output_types
2g
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*#
_output_shapes
:?????????h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: }
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(`
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:?
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*#
_output_shapes
:?????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(i
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 o
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 M
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	Z
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:O
ConstConst*
_output_shapes
:*
dtype0*
valueB: P
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: f
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	X
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: M
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: G
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: K
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: ^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
: T
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
: C
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
: `
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0R
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
: J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @F
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
: b
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype0R
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
: A
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
: R
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
: L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
: V
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
: E
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
: E
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: ?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype0?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
?'
?
__inference_adapt_step_10392
iterator

iterator_1%
add_readvariableop_resource:	 !
readvariableop_resource: #
readvariableop_2_resource: ??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:?????????*"
output_shapes
:?????????*
output_types
2g
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*#
_output_shapes
:?????????h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: }
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(`
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:?
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*#
_output_shapes
:?????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(i
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 o
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 M
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	Z
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:O
ConstConst*
_output_shapes
:*
dtype0*
valueB: P
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: f
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	X
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: M
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: G
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: K
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: ^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
: T
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
: C
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
: `
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0R
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
: J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @F
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
: b
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype0R
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
: A
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
: R
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
: L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
: V
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
: E
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
: E
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: ?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype0?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
?'
?
__inference_adapt_step_9736
iterator

iterator_1%
add_readvariableop_resource:	 !
readvariableop_resource: #
readvariableop_2_resource: ??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:?????????*"
output_shapes
:?????????*
output_types
2g
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*#
_output_shapes
:?????????h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: }
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(`
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:?
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*#
_output_shapes
:?????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(i
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 o
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 M
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	Z
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:O
ConstConst*
_output_shapes
:*
dtype0*
valueB: P
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: f
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	X
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: M
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: G
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: K
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: ^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
: T
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
: C
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
: `
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0R
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
: J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @F
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
: b
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype0R
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
: A
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
: R
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
: L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
: V
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
: E
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
: E
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: ?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype0?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
?0
?
$__inference_model_layer_call_fn_9210
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
	inputs_13
	inputs_14
	inputs_15
	inputs_16
	inputs_17
	inputs_18
	inputs_19
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39: 

unknown_40: 

unknown_41: 

unknown_42:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12	inputs_13	inputs_14	inputs_15	inputs_16	inputs_17	inputs_18	inputs_19unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*K
TinD
B2@*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
<=>?*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_7964o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????::::::::::::::::::::::::::::::::::::::::: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/8:Q	M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/9:R
N
'
_output_shapes
:?????????
#
_user_specified_name	inputs/10:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/11:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/12:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/13:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/14:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/15:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/16:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/17:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/18:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/19: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::  

_output_shapes
:: !

_output_shapes
:: "

_output_shapes
:: #

_output_shapes
:: $

_output_shapes
:: %

_output_shapes
:: &

_output_shapes
:: '

_output_shapes
:: (

_output_shapes
:: )

_output_shapes
:: *

_output_shapes
:: +

_output_shapes
:: ,

_output_shapes
:: -

_output_shapes
:: .

_output_shapes
:: /

_output_shapes
:: 0

_output_shapes
:: 1

_output_shapes
:: 2

_output_shapes
:: 3

_output_shapes
:: 4

_output_shapes
:: 5

_output_shapes
:: 6

_output_shapes
:: 7

_output_shapes
:: 8

_output_shapes
:: 9

_output_shapes
:: :

_output_shapes
:: ;

_output_shapes
:
?'
?
__inference_adapt_step_10298
iterator

iterator_1%
add_readvariableop_resource:	 !
readvariableop_resource: #
readvariableop_2_resource: ??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:?????????*"
output_shapes
:?????????*
output_types
2g
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*#
_output_shapes
:?????????h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: }
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(`
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:?
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*#
_output_shapes
:?????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(i
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 o
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 M
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	Z
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:O
ConstConst*
_output_shapes
:*
dtype0*
valueB: P
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: f
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	X
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: M
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: G
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: K
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: ^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
: T
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
: C
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
: `
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0R
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
: J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @F
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
: b
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype0R
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
: A
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
: R
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
: L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
: V
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
: E
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
: E
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: ?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype0?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
?'
?
__inference_adapt_step_10345
iterator

iterator_1%
add_readvariableop_resource:	 !
readvariableop_resource: #
readvariableop_2_resource: ??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:?????????*"
output_shapes
:?????????*
output_types
2g
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*#
_output_shapes
:?????????h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: }
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(`
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:?
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*#
_output_shapes
:?????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(i
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 o
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 M
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	Z
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:O
ConstConst*
_output_shapes
:*
dtype0*
valueB: P
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: f
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	X
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: M
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: G
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: K
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: ^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
: T
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
: C
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
: `
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0R
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
: J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @F
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
: b
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype0R
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
: A
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
: R
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
: L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
: V
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
: E
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
: E
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: ?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype0?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
??
?
?__inference_model_layer_call_and_return_conditional_losses_8431

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
	inputs_13
	inputs_14
	inputs_15
	inputs_16
	inputs_17
	inputs_18
	inputs_19
normalization_sub_y
normalization_sqrt_x
normalization_1_sub_y
normalization_1_sqrt_x
normalization_2_sub_y
normalization_2_sqrt_x
normalization_3_sub_y
normalization_3_sqrt_x
normalization_4_sub_y
normalization_4_sqrt_x
normalization_5_sub_y
normalization_5_sqrt_x
normalization_6_sub_y
normalization_6_sqrt_x
normalization_7_sub_y
normalization_7_sqrt_x
normalization_8_sub_y
normalization_8_sqrt_x
normalization_9_sub_y
normalization_9_sqrt_x
normalization_10_sub_y
normalization_10_sqrt_x
normalization_11_sub_y
normalization_11_sqrt_x
normalization_12_sub_y
normalization_12_sqrt_x
normalization_13_sub_y
normalization_13_sqrt_x
normalization_14_sub_y
normalization_14_sqrt_x
normalization_15_sub_y
normalization_15_sqrt_x
normalization_16_sub_y
normalization_16_sqrt_x
normalization_17_sub_y
normalization_17_sqrt_x
normalization_18_sub_y
normalization_18_sqrt_x
normalization_19_sub_y
normalization_19_sqrt_x

dense_8419: 

dense_8421: 
dense_1_8425: 
dense_1_8427:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dropout/StatefulPartitionedCallg
normalization/subSubinputsnormalization_sub_y*
T0*'
_output_shapes
:?????????U
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes
:\
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_1/subSubinputs_1normalization_1_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_1/SqrtSqrtnormalization_1_sqrt_x*
T0*
_output_shapes
:^
normalization_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_1/MaximumMaximumnormalization_1/Sqrt:y:0"normalization_1/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_1/truedivRealDivnormalization_1/sub:z:0normalization_1/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_2/subSubinputs_2normalization_2_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_2/SqrtSqrtnormalization_2_sqrt_x*
T0*
_output_shapes
:^
normalization_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_2/MaximumMaximumnormalization_2/Sqrt:y:0"normalization_2/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_2/truedivRealDivnormalization_2/sub:z:0normalization_2/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_3/subSubinputs_3normalization_3_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_3/SqrtSqrtnormalization_3_sqrt_x*
T0*
_output_shapes
:^
normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_4/subSubinputs_4normalization_4_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_4/SqrtSqrtnormalization_4_sqrt_x*
T0*
_output_shapes
:^
normalization_4/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_4/MaximumMaximumnormalization_4/Sqrt:y:0"normalization_4/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_4/truedivRealDivnormalization_4/sub:z:0normalization_4/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_5/subSubinputs_5normalization_5_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_5/SqrtSqrtnormalization_5_sqrt_x*
T0*
_output_shapes
:^
normalization_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_5/MaximumMaximumnormalization_5/Sqrt:y:0"normalization_5/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_5/truedivRealDivnormalization_5/sub:z:0normalization_5/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_6/subSubinputs_6normalization_6_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_6/SqrtSqrtnormalization_6_sqrt_x*
T0*
_output_shapes
:^
normalization_6/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_6/MaximumMaximumnormalization_6/Sqrt:y:0"normalization_6/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_6/truedivRealDivnormalization_6/sub:z:0normalization_6/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_7/subSubinputs_7normalization_7_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_7/SqrtSqrtnormalization_7_sqrt_x*
T0*
_output_shapes
:^
normalization_7/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_7/MaximumMaximumnormalization_7/Sqrt:y:0"normalization_7/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_7/truedivRealDivnormalization_7/sub:z:0normalization_7/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_8/subSubinputs_8normalization_8_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_8/SqrtSqrtnormalization_8_sqrt_x*
T0*
_output_shapes
:^
normalization_8/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_8/MaximumMaximumnormalization_8/Sqrt:y:0"normalization_8/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_8/truedivRealDivnormalization_8/sub:z:0normalization_8/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_9/subSubinputs_9normalization_9_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_9/SqrtSqrtnormalization_9_sqrt_x*
T0*
_output_shapes
:^
normalization_9/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_9/MaximumMaximumnormalization_9/Sqrt:y:0"normalization_9/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_9/truedivRealDivnormalization_9/sub:z:0normalization_9/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_10/subSub	inputs_10normalization_10_sub_y*
T0*'
_output_shapes
:?????????[
normalization_10/SqrtSqrtnormalization_10_sqrt_x*
T0*
_output_shapes
:_
normalization_10/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_10/MaximumMaximumnormalization_10/Sqrt:y:0#normalization_10/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_10/truedivRealDivnormalization_10/sub:z:0normalization_10/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_11/subSub	inputs_11normalization_11_sub_y*
T0*'
_output_shapes
:?????????[
normalization_11/SqrtSqrtnormalization_11_sqrt_x*
T0*
_output_shapes
:_
normalization_11/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_11/MaximumMaximumnormalization_11/Sqrt:y:0#normalization_11/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_11/truedivRealDivnormalization_11/sub:z:0normalization_11/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_12/subSub	inputs_12normalization_12_sub_y*
T0*'
_output_shapes
:?????????[
normalization_12/SqrtSqrtnormalization_12_sqrt_x*
T0*
_output_shapes
:_
normalization_12/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_12/MaximumMaximumnormalization_12/Sqrt:y:0#normalization_12/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_12/truedivRealDivnormalization_12/sub:z:0normalization_12/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_13/subSub	inputs_13normalization_13_sub_y*
T0*'
_output_shapes
:?????????[
normalization_13/SqrtSqrtnormalization_13_sqrt_x*
T0*
_output_shapes
:_
normalization_13/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_13/MaximumMaximumnormalization_13/Sqrt:y:0#normalization_13/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_13/truedivRealDivnormalization_13/sub:z:0normalization_13/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_14/subSub	inputs_14normalization_14_sub_y*
T0*'
_output_shapes
:?????????[
normalization_14/SqrtSqrtnormalization_14_sqrt_x*
T0*
_output_shapes
:_
normalization_14/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_14/MaximumMaximumnormalization_14/Sqrt:y:0#normalization_14/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_14/truedivRealDivnormalization_14/sub:z:0normalization_14/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_15/subSub	inputs_15normalization_15_sub_y*
T0*'
_output_shapes
:?????????[
normalization_15/SqrtSqrtnormalization_15_sqrt_x*
T0*
_output_shapes
:_
normalization_15/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_15/MaximumMaximumnormalization_15/Sqrt:y:0#normalization_15/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_15/truedivRealDivnormalization_15/sub:z:0normalization_15/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_16/subSub	inputs_16normalization_16_sub_y*
T0*'
_output_shapes
:?????????[
normalization_16/SqrtSqrtnormalization_16_sqrt_x*
T0*
_output_shapes
:_
normalization_16/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_16/MaximumMaximumnormalization_16/Sqrt:y:0#normalization_16/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_16/truedivRealDivnormalization_16/sub:z:0normalization_16/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_17/subSub	inputs_17normalization_17_sub_y*
T0*'
_output_shapes
:?????????[
normalization_17/SqrtSqrtnormalization_17_sqrt_x*
T0*
_output_shapes
:_
normalization_17/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_17/MaximumMaximumnormalization_17/Sqrt:y:0#normalization_17/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_17/truedivRealDivnormalization_17/sub:z:0normalization_17/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_18/subSub	inputs_18normalization_18_sub_y*
T0*'
_output_shapes
:?????????[
normalization_18/SqrtSqrtnormalization_18_sqrt_x*
T0*
_output_shapes
:_
normalization_18/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_18/MaximumMaximumnormalization_18/Sqrt:y:0#normalization_18/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_18/truedivRealDivnormalization_18/sub:z:0normalization_18/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_19/subSub	inputs_19normalization_19_sub_y*
T0*'
_output_shapes
:?????????[
normalization_19/SqrtSqrtnormalization_19_sqrt_x*
T0*
_output_shapes
:_
normalization_19/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_19/MaximumMaximumnormalization_19/Sqrt:y:0#normalization_19/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_19/truedivRealDivnormalization_19/sub:z:0normalization_19/Maximum:z:0*
T0*'
_output_shapes
:??????????
concatenate/PartitionedCallPartitionedCallnormalization/truediv:z:0normalization_1/truediv:z:0normalization_2/truediv:z:0normalization_3/truediv:z:0normalization_4/truediv:z:0normalization_5/truediv:z:0normalization_6/truediv:z:0normalization_7/truediv:z:0normalization_8/truediv:z:0normalization_9/truediv:z:0normalization_10/truediv:z:0normalization_11/truediv:z:0normalization_12/truediv:z:0normalization_13/truediv:z:0normalization_14/truediv:z:0normalization_15/truediv:z:0normalization_16/truediv:z:0normalization_17/truediv:z:0normalization_18/truediv:z:0normalization_19/truediv:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_7920?
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0
dense_8419
dense_8421*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_7933?
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_8085?
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_8425dense_1_8427*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_7957w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????::::::::::::::::::::::::::::::::::::::::: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O	K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O
K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::  

_output_shapes
:: !

_output_shapes
:: "

_output_shapes
:: #

_output_shapes
:: $

_output_shapes
:: %

_output_shapes
:: &

_output_shapes
:: '

_output_shapes
:: (

_output_shapes
:: )

_output_shapes
:: *

_output_shapes
:: +

_output_shapes
:: ,

_output_shapes
:: -

_output_shapes
:: .

_output_shapes
:: /

_output_shapes
:: 0

_output_shapes
:: 1

_output_shapes
:: 2

_output_shapes
:: 3

_output_shapes
:: 4

_output_shapes
:: 5

_output_shapes
:: 6

_output_shapes
:: 7

_output_shapes
:: 8

_output_shapes
:: 9

_output_shapes
:: :

_output_shapes
:: ;

_output_shapes
:
?'
?
__inference_adapt_step_10627
iterator

iterator_1%
add_readvariableop_resource:	 !
readvariableop_resource: #
readvariableop_2_resource: ??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:?????????*"
output_shapes
:?????????*
output_types
2g
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*#
_output_shapes
:?????????h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: }
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(`
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:?
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*#
_output_shapes
:?????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(i
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 o
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 M
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	Z
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:O
ConstConst*
_output_shapes
:*
dtype0*
valueB: P
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: f
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	X
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: M
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: G
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: K
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: ^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
: T
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
: C
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
: `
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0R
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
: J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @F
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
: b
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype0R
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
: A
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
: R
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
: L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
: V
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
: E
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
: E
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: ?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype0?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
?

?
A__inference_dense_1_layer_call_and_return_conditional_losses_7957

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
`
A__inference_dropout_layer_call_and_return_conditional_losses_8085

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:????????? C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:????????? Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
+__inference_concatenate_layer_call_fn_10651
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
	inputs_13
	inputs_14
	inputs_15
	inputs_16
	inputs_17
	inputs_18
	inputs_19
identity?
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12	inputs_13	inputs_14	inputs_15	inputs_16	inputs_17	inputs_18	inputs_19*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_7920`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/8:Q	M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/9:R
N
'
_output_shapes
:?????????
#
_user_specified_name	inputs/10:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/11:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/12:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/13:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/14:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/15:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/16:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/17:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/18:RN
'
_output_shapes
:?????????
#
_user_specified_name	inputs/19
?
_
A__inference_dropout_layer_call_and_return_conditional_losses_7944

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:????????? [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:????????? "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_10711

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:????????? [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:????????? "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs

?
?__inference_model_layer_call_and_return_conditional_losses_7964

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
	inputs_13
	inputs_14
	inputs_15
	inputs_16
	inputs_17
	inputs_18
	inputs_19
normalization_sub_y
normalization_sqrt_x
normalization_1_sub_y
normalization_1_sqrt_x
normalization_2_sub_y
normalization_2_sqrt_x
normalization_3_sub_y
normalization_3_sqrt_x
normalization_4_sub_y
normalization_4_sqrt_x
normalization_5_sub_y
normalization_5_sqrt_x
normalization_6_sub_y
normalization_6_sqrt_x
normalization_7_sub_y
normalization_7_sqrt_x
normalization_8_sub_y
normalization_8_sqrt_x
normalization_9_sub_y
normalization_9_sqrt_x
normalization_10_sub_y
normalization_10_sqrt_x
normalization_11_sub_y
normalization_11_sqrt_x
normalization_12_sub_y
normalization_12_sqrt_x
normalization_13_sub_y
normalization_13_sqrt_x
normalization_14_sub_y
normalization_14_sqrt_x
normalization_15_sub_y
normalization_15_sqrt_x
normalization_16_sub_y
normalization_16_sqrt_x
normalization_17_sub_y
normalization_17_sqrt_x
normalization_18_sub_y
normalization_18_sqrt_x
normalization_19_sub_y
normalization_19_sqrt_x

dense_7934: 

dense_7936: 
dense_1_7958: 
dense_1_7960:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCallg
normalization/subSubinputsnormalization_sub_y*
T0*'
_output_shapes
:?????????U
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes
:\
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_1/subSubinputs_1normalization_1_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_1/SqrtSqrtnormalization_1_sqrt_x*
T0*
_output_shapes
:^
normalization_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_1/MaximumMaximumnormalization_1/Sqrt:y:0"normalization_1/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_1/truedivRealDivnormalization_1/sub:z:0normalization_1/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_2/subSubinputs_2normalization_2_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_2/SqrtSqrtnormalization_2_sqrt_x*
T0*
_output_shapes
:^
normalization_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_2/MaximumMaximumnormalization_2/Sqrt:y:0"normalization_2/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_2/truedivRealDivnormalization_2/sub:z:0normalization_2/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_3/subSubinputs_3normalization_3_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_3/SqrtSqrtnormalization_3_sqrt_x*
T0*
_output_shapes
:^
normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_4/subSubinputs_4normalization_4_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_4/SqrtSqrtnormalization_4_sqrt_x*
T0*
_output_shapes
:^
normalization_4/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_4/MaximumMaximumnormalization_4/Sqrt:y:0"normalization_4/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_4/truedivRealDivnormalization_4/sub:z:0normalization_4/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_5/subSubinputs_5normalization_5_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_5/SqrtSqrtnormalization_5_sqrt_x*
T0*
_output_shapes
:^
normalization_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_5/MaximumMaximumnormalization_5/Sqrt:y:0"normalization_5/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_5/truedivRealDivnormalization_5/sub:z:0normalization_5/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_6/subSubinputs_6normalization_6_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_6/SqrtSqrtnormalization_6_sqrt_x*
T0*
_output_shapes
:^
normalization_6/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_6/MaximumMaximumnormalization_6/Sqrt:y:0"normalization_6/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_6/truedivRealDivnormalization_6/sub:z:0normalization_6/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_7/subSubinputs_7normalization_7_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_7/SqrtSqrtnormalization_7_sqrt_x*
T0*
_output_shapes
:^
normalization_7/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_7/MaximumMaximumnormalization_7/Sqrt:y:0"normalization_7/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_7/truedivRealDivnormalization_7/sub:z:0normalization_7/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_8/subSubinputs_8normalization_8_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_8/SqrtSqrtnormalization_8_sqrt_x*
T0*
_output_shapes
:^
normalization_8/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_8/MaximumMaximumnormalization_8/Sqrt:y:0"normalization_8/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_8/truedivRealDivnormalization_8/sub:z:0normalization_8/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_9/subSubinputs_9normalization_9_sub_y*
T0*'
_output_shapes
:?????????Y
normalization_9/SqrtSqrtnormalization_9_sqrt_x*
T0*
_output_shapes
:^
normalization_9/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_9/MaximumMaximumnormalization_9/Sqrt:y:0"normalization_9/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_9/truedivRealDivnormalization_9/sub:z:0normalization_9/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_10/subSub	inputs_10normalization_10_sub_y*
T0*'
_output_shapes
:?????????[
normalization_10/SqrtSqrtnormalization_10_sqrt_x*
T0*
_output_shapes
:_
normalization_10/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_10/MaximumMaximumnormalization_10/Sqrt:y:0#normalization_10/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_10/truedivRealDivnormalization_10/sub:z:0normalization_10/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_11/subSub	inputs_11normalization_11_sub_y*
T0*'
_output_shapes
:?????????[
normalization_11/SqrtSqrtnormalization_11_sqrt_x*
T0*
_output_shapes
:_
normalization_11/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_11/MaximumMaximumnormalization_11/Sqrt:y:0#normalization_11/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_11/truedivRealDivnormalization_11/sub:z:0normalization_11/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_12/subSub	inputs_12normalization_12_sub_y*
T0*'
_output_shapes
:?????????[
normalization_12/SqrtSqrtnormalization_12_sqrt_x*
T0*
_output_shapes
:_
normalization_12/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_12/MaximumMaximumnormalization_12/Sqrt:y:0#normalization_12/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_12/truedivRealDivnormalization_12/sub:z:0normalization_12/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_13/subSub	inputs_13normalization_13_sub_y*
T0*'
_output_shapes
:?????????[
normalization_13/SqrtSqrtnormalization_13_sqrt_x*
T0*
_output_shapes
:_
normalization_13/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_13/MaximumMaximumnormalization_13/Sqrt:y:0#normalization_13/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_13/truedivRealDivnormalization_13/sub:z:0normalization_13/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_14/subSub	inputs_14normalization_14_sub_y*
T0*'
_output_shapes
:?????????[
normalization_14/SqrtSqrtnormalization_14_sqrt_x*
T0*
_output_shapes
:_
normalization_14/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_14/MaximumMaximumnormalization_14/Sqrt:y:0#normalization_14/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_14/truedivRealDivnormalization_14/sub:z:0normalization_14/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_15/subSub	inputs_15normalization_15_sub_y*
T0*'
_output_shapes
:?????????[
normalization_15/SqrtSqrtnormalization_15_sqrt_x*
T0*
_output_shapes
:_
normalization_15/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_15/MaximumMaximumnormalization_15/Sqrt:y:0#normalization_15/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_15/truedivRealDivnormalization_15/sub:z:0normalization_15/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_16/subSub	inputs_16normalization_16_sub_y*
T0*'
_output_shapes
:?????????[
normalization_16/SqrtSqrtnormalization_16_sqrt_x*
T0*
_output_shapes
:_
normalization_16/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_16/MaximumMaximumnormalization_16/Sqrt:y:0#normalization_16/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_16/truedivRealDivnormalization_16/sub:z:0normalization_16/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_17/subSub	inputs_17normalization_17_sub_y*
T0*'
_output_shapes
:?????????[
normalization_17/SqrtSqrtnormalization_17_sqrt_x*
T0*
_output_shapes
:_
normalization_17/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_17/MaximumMaximumnormalization_17/Sqrt:y:0#normalization_17/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_17/truedivRealDivnormalization_17/sub:z:0normalization_17/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_18/subSub	inputs_18normalization_18_sub_y*
T0*'
_output_shapes
:?????????[
normalization_18/SqrtSqrtnormalization_18_sqrt_x*
T0*
_output_shapes
:_
normalization_18/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_18/MaximumMaximumnormalization_18/Sqrt:y:0#normalization_18/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_18/truedivRealDivnormalization_18/sub:z:0normalization_18/Maximum:z:0*
T0*'
_output_shapes
:?????????p
normalization_19/subSub	inputs_19normalization_19_sub_y*
T0*'
_output_shapes
:?????????[
normalization_19/SqrtSqrtnormalization_19_sqrt_x*
T0*
_output_shapes
:_
normalization_19/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_19/MaximumMaximumnormalization_19/Sqrt:y:0#normalization_19/Maximum/y:output:0*
T0*
_output_shapes
:?
normalization_19/truedivRealDivnormalization_19/sub:z:0normalization_19/Maximum:z:0*
T0*'
_output_shapes
:??????????
concatenate/PartitionedCallPartitionedCallnormalization/truediv:z:0normalization_1/truediv:z:0normalization_2/truediv:z:0normalization_3/truediv:z:0normalization_4/truediv:z:0normalization_5/truediv:z:0normalization_6/truediv:z:0normalization_7/truediv:z:0normalization_8/truediv:z:0normalization_9/truediv:z:0normalization_10/truediv:z:0normalization_11/truediv:z:0normalization_12/truediv:z:0normalization_13/truediv:z:0normalization_14/truediv:z:0normalization_15/truediv:z:0normalization_16/truediv:z:0normalization_17/truediv:z:0normalization_18/truediv:z:0normalization_19/truediv:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_7920?
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0
dense_7934
dense_7936*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_7933?
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_7944?
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_7958dense_1_7960*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_7957w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????::::::::::::::::::::::::::::::::::::::::: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O	K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O
K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::  

_output_shapes
:: !

_output_shapes
:: "

_output_shapes
:: #

_output_shapes
:: $

_output_shapes
:: %

_output_shapes
:: &

_output_shapes
:: '

_output_shapes
:: (

_output_shapes
:: )

_output_shapes
:: *

_output_shapes
:: +

_output_shapes
:: ,

_output_shapes
:: -

_output_shapes
:: .

_output_shapes
:: /

_output_shapes
:: 0

_output_shapes
:: 1

_output_shapes
:: 2

_output_shapes
:: 3

_output_shapes
:: 4

_output_shapes
:: 5

_output_shapes
:: 6

_output_shapes
:: 7

_output_shapes
:: 8

_output_shapes
:: 9

_output_shapes
:: :

_output_shapes
:: ;

_output_shapes
:"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?

serving_default?

G
battery_power6
serving_default_battery_power:0?????????
5
blue-
serving_default_blue:0?????????
C
clock_speed4
serving_default_clock_speed:0?????????
=
dual_sim1
serving_default_dual_sim:0?????????
1
fc+
serving_default_fc:0?????????
9
four_g/
serving_default_four_g:0?????????
A

int_memory3
serving_default_int_memory:0?????????
7
m_dep.
serving_default_m_dep:0?????????
?
	mobile_wt2
serving_default_mobile_wt:0?????????
;
n_cores0
serving_default_n_cores:0?????????
1
pc+
serving_default_pc:0?????????
?
	px_height2
serving_default_px_height:0?????????
=
px_width1
serving_default_px_width:0?????????
3
ram,
serving_default_ram:0?????????
5
sc_h-
serving_default_sc_h:0?????????
5
sc_w-
serving_default_sc_w:0?????????
?
	talk_time2
serving_default_talk_time:0?????????
;
three_g0
serving_default_three_g:0?????????
E
touch_screen5
serving_default_touch_screen:0?????????
5
wifi-
serving_default_wifi:0?????????;
dense_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer_with_weights-0
layer-20
layer_with_weights-1
layer-21
layer_with_weights-2
layer-22
layer_with_weights-3
layer-23
layer_with_weights-4
layer-24
layer_with_weights-5
layer-25
layer_with_weights-6
layer-26
layer_with_weights-7
layer-27
layer_with_weights-8
layer-28
layer_with_weights-9
layer-29
layer_with_weights-10
layer-30
 layer_with_weights-11
 layer-31
!layer_with_weights-12
!layer-32
"layer_with_weights-13
"layer-33
#layer_with_weights-14
#layer-34
$layer_with_weights-15
$layer-35
%layer_with_weights-16
%layer-36
&layer_with_weights-17
&layer-37
'layer_with_weights-18
'layer-38
(layer_with_weights-19
(layer-39
)layer-40
*layer_with_weights-20
*layer-41
+layer-42
,layer_with_weights-21
,layer-43
-	optimizer
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2
signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
?
3
_keep_axis
4_reduce_axis
5_reduce_axis_mask
6_broadcast_shape
7mean
7
adapt_mean
8variance
8adapt_variance
	9count
:	keras_api
?_adapt_function"
_tf_keras_layer
?
;
_keep_axis
<_reduce_axis
=_reduce_axis_mask
>_broadcast_shape
?mean
?
adapt_mean
@variance
@adapt_variance
	Acount
B	keras_api
?_adapt_function"
_tf_keras_layer
?
C
_keep_axis
D_reduce_axis
E_reduce_axis_mask
F_broadcast_shape
Gmean
G
adapt_mean
Hvariance
Hadapt_variance
	Icount
J	keras_api
?_adapt_function"
_tf_keras_layer
?
K
_keep_axis
L_reduce_axis
M_reduce_axis_mask
N_broadcast_shape
Omean
O
adapt_mean
Pvariance
Padapt_variance
	Qcount
R	keras_api
?_adapt_function"
_tf_keras_layer
?
S
_keep_axis
T_reduce_axis
U_reduce_axis_mask
V_broadcast_shape
Wmean
W
adapt_mean
Xvariance
Xadapt_variance
	Ycount
Z	keras_api
?_adapt_function"
_tf_keras_layer
?
[
_keep_axis
\_reduce_axis
]_reduce_axis_mask
^_broadcast_shape
_mean
_
adapt_mean
`variance
`adapt_variance
	acount
b	keras_api
?_adapt_function"
_tf_keras_layer
?
c
_keep_axis
d_reduce_axis
e_reduce_axis_mask
f_broadcast_shape
gmean
g
adapt_mean
hvariance
hadapt_variance
	icount
j	keras_api
?_adapt_function"
_tf_keras_layer
?
k
_keep_axis
l_reduce_axis
m_reduce_axis_mask
n_broadcast_shape
omean
o
adapt_mean
pvariance
padapt_variance
	qcount
r	keras_api
?_adapt_function"
_tf_keras_layer
?
s
_keep_axis
t_reduce_axis
u_reduce_axis_mask
v_broadcast_shape
wmean
w
adapt_mean
xvariance
xadapt_variance
	ycount
z	keras_api
?_adapt_function"
_tf_keras_layer
?
{
_keep_axis
|_reduce_axis
}_reduce_axis_mask
~_broadcast_shape
mean

adapt_mean
?variance
?adapt_variance

?count
?	keras_api
?_adapt_function"
_tf_keras_layer
?
?
_keep_axis
?_reduce_axis
?_reduce_axis_mask
?_broadcast_shape
	?mean
?
adapt_mean
?variance
?adapt_variance

?count
?	keras_api
?_adapt_function"
_tf_keras_layer
?
?
_keep_axis
?_reduce_axis
?_reduce_axis_mask
?_broadcast_shape
	?mean
?
adapt_mean
?variance
?adapt_variance

?count
?	keras_api
?_adapt_function"
_tf_keras_layer
?
?
_keep_axis
?_reduce_axis
?_reduce_axis_mask
?_broadcast_shape
	?mean
?
adapt_mean
?variance
?adapt_variance

?count
?	keras_api
?_adapt_function"
_tf_keras_layer
?
?
_keep_axis
?_reduce_axis
?_reduce_axis_mask
?_broadcast_shape
	?mean
?
adapt_mean
?variance
?adapt_variance

?count
?	keras_api
?_adapt_function"
_tf_keras_layer
?
?
_keep_axis
?_reduce_axis
?_reduce_axis_mask
?_broadcast_shape
	?mean
?
adapt_mean
?variance
?adapt_variance

?count
?	keras_api
?_adapt_function"
_tf_keras_layer
?
?
_keep_axis
?_reduce_axis
?_reduce_axis_mask
?_broadcast_shape
	?mean
?
adapt_mean
?variance
?adapt_variance

?count
?	keras_api
?_adapt_function"
_tf_keras_layer
?
?
_keep_axis
?_reduce_axis
?_reduce_axis_mask
?_broadcast_shape
	?mean
?
adapt_mean
?variance
?adapt_variance

?count
?	keras_api
?_adapt_function"
_tf_keras_layer
?
?
_keep_axis
?_reduce_axis
?_reduce_axis_mask
?_broadcast_shape
	?mean
?
adapt_mean
?variance
?adapt_variance

?count
?	keras_api
?_adapt_function"
_tf_keras_layer
?
?
_keep_axis
?_reduce_axis
?_reduce_axis_mask
?_broadcast_shape
	?mean
?
adapt_mean
?variance
?adapt_variance

?count
?	keras_api
?_adapt_function"
_tf_keras_layer
?
?
_keep_axis
?_reduce_axis
?_reduce_axis_mask
?_broadcast_shape
	?mean
?
adapt_mean
?variance
?adapt_variance

?count
?	keras_api
?_adapt_function"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
"
	optimizer
?
70
81
92
?3
@4
A5
G6
H7
I8
O9
P10
Q11
W12
X13
Y14
_15
`16
a17
g18
h19
i20
o21
p22
q23
w24
x25
y26
27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?50
?51
?52
?53
?54
?55
?56
?57
?58
?59
?60
?61
?62
?63"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
.	variables
/trainable_variables
0regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
 2mean
: 2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
 2mean
: 2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
 2mean
: 2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
 2mean
: 2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
 2mean
: 2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
 2mean
: 2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
 2mean
: 2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
 2mean
: 2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
 2mean
: 2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
 2mean
: 2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
 2mean
: 2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
 2mean
: 2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
 2mean
: 2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
 2mean
: 2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
 2mean
: 2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
 2mean
: 2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
 2mean
: 2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
 2mean
: 2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
 2mean
: 2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
 2mean
: 2variance
:	 2count
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
: 2dense/kernel
: 2
dense/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 : 2dense_1/kernel
:2dense_1/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
70
81
92
?3
@4
A5
G6
H7
I8
O9
P10
Q11
W12
X13
Y14
_15
`16
a17
g18
h19
i20
o21
p22
q23
w24
x25
y26
27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?50
?51
?52
?53
?54
?55
?56
?57
?58
?59"
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
?2?
$__inference_model_layer_call_fn_8055
$__inference_model_layer_call_fn_9210
$__inference_model_layer_call_fn_9322
$__inference_model_layer_call_fn_8634?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
?__inference_model_layer_call_and_return_conditional_losses_9502
?__inference_model_layer_call_and_return_conditional_losses_9689
?__inference_model_layer_call_and_return_conditional_losses_8809
?__inference_model_layer_call_and_return_conditional_losses_8984?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
__inference__wrapped_model_7710battery_powerblueclock_speeddual_simfcfour_g
int_memorym_dep	mobile_wtn_corespc	px_heightpx_widthramsc_hsc_w	talk_timethree_gtouch_screenwifi"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_9736?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_9783?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_9829?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_9876?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_9923?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_9970?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_10017?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_10063?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_10110?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_10157?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_10204?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_10251?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_10298?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_10345?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_10392?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_10439?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_10486?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_10533?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_10580?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_10627?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_concatenate_layer_call_fn_10651?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_concatenate_layer_call_and_return_conditional_losses_10676?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_dense_layer_call_fn_10685?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_dense_layer_call_and_return_conditional_losses_10696?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dropout_layer_call_fn_10701
'__inference_dropout_layer_call_fn_10706?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_dropout_layer_call_and_return_conditional_losses_10711
B__inference_dropout_layer_call_and_return_conditional_losses_10723?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_dense_1_layer_call_fn_10732?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_1_layer_call_and_return_conditional_losses_10743?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
"__inference_signature_wrapper_9098battery_powerblueclock_speeddual_simfcfour_g
int_memorym_dep	mobile_wtn_corespc	px_heightpx_widthramsc_hsc_w	talk_timethree_gtouch_screenwifi"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4
J	
Const_5
J	
Const_6
J	
Const_7
J	
Const_8
J	
Const_9
J

Const_10
J

Const_11
J

Const_12
J

Const_13
J

Const_14
J

Const_15
J

Const_16
J

Const_17
J

Const_18
J

Const_19
J

Const_20
J

Const_21
J

Const_22
J

Const_23
J

Const_24
J

Const_25
J

Const_26
J

Const_27
J

Const_28
J

Const_29
J

Const_30
J

Const_31
J

Const_32
J

Const_33
J

Const_34
J

Const_35
J

Const_36
J

Const_37
J

Const_38
J

Const_39?
__inference__wrapped_model_7710?X???????????????????????????????????????????????
???
???
'?$
battery_power?????????
?
blue?????????
%?"
clock_speed?????????
"?
dual_sim?????????
?
fc?????????
 ?
four_g?????????
$?!

int_memory?????????
?
m_dep?????????
#? 
	mobile_wt?????????
!?
n_cores?????????
?
pc?????????
#? 
	px_height?????????
"?
px_width?????????
?
ram?????????
?
sc_h?????????
?
sc_w?????????
#? 
	talk_time?????????
!?
three_g?????????
&?#
touch_screen?????????
?
wifi?????????
? "1?.
,
dense_1!?
dense_1?????????j
__inference_adapt_step_10017Jigh??<
5?2
0?-?
??????????IteratorSpec 
? "
 j
__inference_adapt_step_10063Jqop??<
5?2
0?-?
??????????IteratorSpec 
? "
 j
__inference_adapt_step_10110Jywx??<
5?2
0?-?
??????????IteratorSpec 
? "
 l
__inference_adapt_step_10157L????<
5?2
0?-?
??????????IteratorSpec 
? "
 m
__inference_adapt_step_10204M?????<
5?2
0?-?
??????????IteratorSpec 
? "
 m
__inference_adapt_step_10251M?????<
5?2
0?-?
??????????IteratorSpec 
? "
 m
__inference_adapt_step_10298M?????<
5?2
0?-?
??????????IteratorSpec 
? "
 m
__inference_adapt_step_10345M?????<
5?2
0?-?
??????????IteratorSpec 
? "
 m
__inference_adapt_step_10392M?????<
5?2
0?-?
??????????IteratorSpec 
? "
 m
__inference_adapt_step_10439M?????<
5?2
0?-?
??????????IteratorSpec 
? "
 m
__inference_adapt_step_10486M?????<
5?2
0?-?
??????????IteratorSpec 
? "
 m
__inference_adapt_step_10533M?????<
5?2
0?-?
??????????IteratorSpec 
? "
 m
__inference_adapt_step_10580M?????<
5?2
0?-?
??????????IteratorSpec 
? "
 m
__inference_adapt_step_10627M?????<
5?2
0?-?
??????????IteratorSpec 
? "
 i
__inference_adapt_step_9736J978??<
5?2
0?-?
??????????IteratorSpec 
? "
 i
__inference_adapt_step_9783JA?@??<
5?2
0?-?
??????????IteratorSpec 
? "
 i
__inference_adapt_step_9829JIGH??<
5?2
0?-?
??????????IteratorSpec 
? "
 i
__inference_adapt_step_9876JQOP??<
5?2
0?-?
??????????IteratorSpec 
? "
 i
__inference_adapt_step_9923JYWX??<
5?2
0?-?
??????????IteratorSpec 
? "
 i
__inference_adapt_step_9970Ja_`??<
5?2
0?-?
??????????IteratorSpec 
? "
 ?
F__inference_concatenate_layer_call_and_return_conditional_losses_10676????
???
???
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????
"?
inputs/4?????????
"?
inputs/5?????????
"?
inputs/6?????????
"?
inputs/7?????????
"?
inputs/8?????????
"?
inputs/9?????????
#? 
	inputs/10?????????
#? 
	inputs/11?????????
#? 
	inputs/12?????????
#? 
	inputs/13?????????
#? 
	inputs/14?????????
#? 
	inputs/15?????????
#? 
	inputs/16?????????
#? 
	inputs/17?????????
#? 
	inputs/18?????????
#? 
	inputs/19?????????
? "%?"
?
0?????????
? ?
+__inference_concatenate_layer_call_fn_10651????
???
???
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????
"?
inputs/4?????????
"?
inputs/5?????????
"?
inputs/6?????????
"?
inputs/7?????????
"?
inputs/8?????????
"?
inputs/9?????????
#? 
	inputs/10?????????
#? 
	inputs/11?????????
#? 
	inputs/12?????????
#? 
	inputs/13?????????
#? 
	inputs/14?????????
#? 
	inputs/15?????????
#? 
	inputs/16?????????
#? 
	inputs/17?????????
#? 
	inputs/18?????????
#? 
	inputs/19?????????
? "???????????
B__inference_dense_1_layer_call_and_return_conditional_losses_10743^??/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? |
'__inference_dense_1_layer_call_fn_10732Q??/?,
%?"
 ?
inputs????????? 
? "???????????
@__inference_dense_layer_call_and_return_conditional_losses_10696^??/?,
%?"
 ?
inputs?????????
? "%?"
?
0????????? 
? z
%__inference_dense_layer_call_fn_10685Q??/?,
%?"
 ?
inputs?????????
? "?????????? ?
B__inference_dropout_layer_call_and_return_conditional_losses_10711\3?0
)?&
 ?
inputs????????? 
p 
? "%?"
?
0????????? 
? ?
B__inference_dropout_layer_call_and_return_conditional_losses_10723\3?0
)?&
 ?
inputs????????? 
p
? "%?"
?
0????????? 
? z
'__inference_dropout_layer_call_fn_10701O3?0
)?&
 ?
inputs????????? 
p 
? "?????????? z
'__inference_dropout_layer_call_fn_10706O3?0
)?&
 ?
inputs????????? 
p
? "?????????? ?
?__inference_model_layer_call_and_return_conditional_losses_8809?X???????????????????????????????????????????????
???
???
'?$
battery_power?????????
?
blue?????????
%?"
clock_speed?????????
"?
dual_sim?????????
?
fc?????????
 ?
four_g?????????
$?!

int_memory?????????
?
m_dep?????????
#? 
	mobile_wt?????????
!?
n_cores?????????
?
pc?????????
#? 
	px_height?????????
"?
px_width?????????
?
ram?????????
?
sc_h?????????
?
sc_w?????????
#? 
	talk_time?????????
!?
three_g?????????
&?#
touch_screen?????????
?
wifi?????????
p 

 
? "%?"
?
0?????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_8984?X???????????????????????????????????????????????
???
???
'?$
battery_power?????????
?
blue?????????
%?"
clock_speed?????????
"?
dual_sim?????????
?
fc?????????
 ?
four_g?????????
$?!

int_memory?????????
?
m_dep?????????
#? 
	mobile_wt?????????
!?
n_cores?????????
?
pc?????????
#? 
	px_height?????????
"?
px_width?????????
?
ram?????????
?
sc_h?????????
?
sc_w?????????
#? 
	talk_time?????????
!?
three_g?????????
&?#
touch_screen?????????
?
wifi?????????
p

 
? "%?"
?
0?????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_9502?X???????????????????????????????????????????????
???
???
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????
"?
inputs/4?????????
"?
inputs/5?????????
"?
inputs/6?????????
"?
inputs/7?????????
"?
inputs/8?????????
"?
inputs/9?????????
#? 
	inputs/10?????????
#? 
	inputs/11?????????
#? 
	inputs/12?????????
#? 
	inputs/13?????????
#? 
	inputs/14?????????
#? 
	inputs/15?????????
#? 
	inputs/16?????????
#? 
	inputs/17?????????
#? 
	inputs/18?????????
#? 
	inputs/19?????????
p 

 
? "%?"
?
0?????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_9689?X???????????????????????????????????????????????
???
???
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????
"?
inputs/4?????????
"?
inputs/5?????????
"?
inputs/6?????????
"?
inputs/7?????????
"?
inputs/8?????????
"?
inputs/9?????????
#? 
	inputs/10?????????
#? 
	inputs/11?????????
#? 
	inputs/12?????????
#? 
	inputs/13?????????
#? 
	inputs/14?????????
#? 
	inputs/15?????????
#? 
	inputs/16?????????
#? 
	inputs/17?????????
#? 
	inputs/18?????????
#? 
	inputs/19?????????
p

 
? "%?"
?
0?????????
? ?
$__inference_model_layer_call_fn_8055?X???????????????????????????????????????????????
???
???
'?$
battery_power?????????
?
blue?????????
%?"
clock_speed?????????
"?
dual_sim?????????
?
fc?????????
 ?
four_g?????????
$?!

int_memory?????????
?
m_dep?????????
#? 
	mobile_wt?????????
!?
n_cores?????????
?
pc?????????
#? 
	px_height?????????
"?
px_width?????????
?
ram?????????
?
sc_h?????????
?
sc_w?????????
#? 
	talk_time?????????
!?
three_g?????????
&?#
touch_screen?????????
?
wifi?????????
p 

 
? "???????????
$__inference_model_layer_call_fn_8634?X???????????????????????????????????????????????
???
???
'?$
battery_power?????????
?
blue?????????
%?"
clock_speed?????????
"?
dual_sim?????????
?
fc?????????
 ?
four_g?????????
$?!

int_memory?????????
?
m_dep?????????
#? 
	mobile_wt?????????
!?
n_cores?????????
?
pc?????????
#? 
	px_height?????????
"?
px_width?????????
?
ram?????????
?
sc_h?????????
?
sc_w?????????
#? 
	talk_time?????????
!?
three_g?????????
&?#
touch_screen?????????
?
wifi?????????
p

 
? "???????????
$__inference_model_layer_call_fn_9210?X???????????????????????????????????????????????
???
???
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????
"?
inputs/4?????????
"?
inputs/5?????????
"?
inputs/6?????????
"?
inputs/7?????????
"?
inputs/8?????????
"?
inputs/9?????????
#? 
	inputs/10?????????
#? 
	inputs/11?????????
#? 
	inputs/12?????????
#? 
	inputs/13?????????
#? 
	inputs/14?????????
#? 
	inputs/15?????????
#? 
	inputs/16?????????
#? 
	inputs/17?????????
#? 
	inputs/18?????????
#? 
	inputs/19?????????
p 

 
? "???????????
$__inference_model_layer_call_fn_9322?X???????????????????????????????????????????????
???
???
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????
"?
inputs/4?????????
"?
inputs/5?????????
"?
inputs/6?????????
"?
inputs/7?????????
"?
inputs/8?????????
"?
inputs/9?????????
#? 
	inputs/10?????????
#? 
	inputs/11?????????
#? 
	inputs/12?????????
#? 
	inputs/13?????????
#? 
	inputs/14?????????
#? 
	inputs/15?????????
#? 
	inputs/16?????????
#? 
	inputs/17?????????
#? 
	inputs/18?????????
#? 
	inputs/19?????????
p

 
? "???????????
"__inference_signature_wrapper_9098?X???????????????????????????????????????????????
? 
???
8
battery_power'?$
battery_power?????????
&
blue?
blue?????????
4
clock_speed%?"
clock_speed?????????
.
dual_sim"?
dual_sim?????????
"
fc?
fc?????????
*
four_g ?
four_g?????????
2

int_memory$?!

int_memory?????????
(
m_dep?
m_dep?????????
0
	mobile_wt#? 
	mobile_wt?????????
,
n_cores!?
n_cores?????????
"
pc?
pc?????????
0
	px_height#? 
	px_height?????????
.
px_width"?
px_width?????????
$
ram?
ram?????????
&
sc_h?
sc_h?????????
&
sc_w?
sc_w?????????
0
	talk_time#? 
	talk_time?????????
,
three_g!?
three_g?????????
6
touch_screen&?#
touch_screen?????????
&
wifi?
wifi?????????"1?.
,
dense_1!?
dense_1?????????