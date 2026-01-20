; ModuleID = '__compute_module_part_00'
source_filename = "__compute_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_KernelThreadDim = type { i64, i64, i64 }
%XLA_CPU_KernelThread = type { i64, i64, i64 }
%XLA_CPU_KernelArg = type { ptr, i64 }

@__llvmsplit_unnamed.14 = private unnamed_addr constant [4 x i8] c"\F3\04\B5?"
@__llvmsplit_unnamed.15 = private unnamed_addr constant [4 x i8] c"\FF\FF\7F\BF"
@__llvmsplit_unnamed.16 = private unnamed_addr constant [4 x i8] c"\00\00\00@"
@__llvmsplit_unnamed.17 = private unnamed_addr constant [4 x i8] c"\00\00\80\BF"
@__llvmsplit_unnamed.18 = private unnamed_addr constant [4 x i8] c"\00\00\80?"
@__llvmsplit_unnamed.19 = private unnamed_addr constant [4 x i8] c"\09\00\00\00"
@__llvmsplit_unnamed.20 = private unnamed_addr constant [4 x i8] c"\00\00@\C0"
@__llvmsplit_unnamed.21 = private unnamed_addr constant [4 x i8] c"\00\00 \C0"
@__llvmsplit_unnamed.22 = private unnamed_addr constant [4 x i8] c"\00\00\A0@"
@__llvmsplit_unnamed.23 = private unnamed_addr constant [4 x i8] c"\9B\F0Q\B9"
@__llvmsplit_unnamed.24 = private unnamed_addr constant [4 x i8] c"\88e\F12"
@__llvmsplit_unnamed.25 = private unnamed_addr constant [4 x i8] c"k\B5\D38"
@__llvmsplit_unnamed.26 = private unnamed_addr constant [4 x i8] c"6K\B84"
@__llvmsplit_unnamed.27 = private unnamed_addr constant [4 x i8] c"r\DC\B0:"
@__llvmsplit_unnamed.28 = private unnamed_addr constant [4 x i8] c"Wsl\B6"
@__llvmsplit_unnamed.29 = private unnamed_addr constant [4 x i8] c"\E7\BDp\BB"
@__llvmsplit_unnamed.30 = private unnamed_addr constant [4 x i8] c"\C1Z\93\B6"
@__llvmsplit_unnamed.31 = private unnamed_addr constant [4 x i8] c"{\12\BC;"
@__llvmsplit_unnamed.32 = private unnamed_addr constant [4 x i8] c"\DB2e9"
@__llvmsplit_unnamed.33 = private unnamed_addr constant [4 x i8] c"\D7\C5\F9\BB"
@__llvmsplit_unnamed.34 = private unnamed_addr constant [4 x i8] c"\08T\A4\BA"
@__llvmsplit_unnamed.35 = private unnamed_addr constant [4 x i8] c"~\A5\1A<"
@__llvmsplit_unnamed.36 = private unnamed_addr constant [4 x i8] c"\EF\E4\88\BB"
@__llvmsplit_unnamed.37 = private unnamed_addr constant [4 x i8] c"\DB6\80?"
@__llvmsplit_unnamed.38 = private unnamed_addr constant [4 x i8] c"c\8F|>"
@__llvmsplit_unnamed.39 = private unnamed_addr constant [4 x i8] c"~O5@"
@__llvmsplit_unnamed.40 = private unnamed_addr constant [4 x i8] c"/.\C0?"
@__llvmsplit_unnamed.41 = private unnamed_addr constant [4 x i8] c"\00\00\80\7F"
@__llvmsplit_unnamed.42 = private unnamed_addr constant [4 x i8] c"\00\00\80?"
@broadcast_multiply_fusion.clone_parallel_bounds = private constant [18 x [2 x [2 x i64]]] [[2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 0, i64 1666]], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 1666, i64 3332]], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 3332, i64 4998]], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 4998, i64 6664]], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 6664, i64 8330]], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 8330, i64 10000]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 2], [2 x i64] [i64 0, i64 1666]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 2], [2 x i64] [i64 1666, i64 3332]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 2], [2 x i64] [i64 3332, i64 4998]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 2], [2 x i64] [i64 4998, i64 6664]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 2], [2 x i64] [i64 6664, i64 8330]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 2], [2 x i64] [i64 8330, i64 10000]], [2 x [2 x i64]] [[2 x i64] [i64 2, i64 3], [2 x i64] [i64 0, i64 1666]], [2 x [2 x i64]] [[2 x i64] [i64 2, i64 3], [2 x i64] [i64 1666, i64 3332]], [2 x [2 x i64]] [[2 x i64] [i64 2, i64 3], [2 x i64] [i64 3332, i64 4998]], [2 x [2 x i64]] [[2 x i64] [i64 2, i64 3], [2 x i64] [i64 4998, i64 6664]], [2 x [2 x i64]] [[2 x i64] [i64 2, i64 3], [2 x i64] [i64 6664, i64 8330]], [2 x [2 x i64]] [[2 x i64] [i64 2, i64 3], [2 x i64] [i64 8330, i64 10000]]]

; Function Attrs: uwtable
define ptr @broadcast_multiply_fusion.clone(ptr %0) #0 {
  %broadcast_multiply_fusion.clone.invar_address.dim.2 = alloca i64, align 8
  %broadcast_multiply_fusion.clone.invar_address.dim.1 = alloca i64, align 8
  %broadcast_multiply_fusion.clone.invar_address.dim.0 = alloca i64, align 8
  %tdims_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 0
  %tdims = load ptr, ptr %tdims_gep, align 8
  %tdim_x_gep = getelementptr inbounds nuw %XLA_CPU_KernelThreadDim, ptr %tdims, i32 0, i32 0
  %tdim_y_gep = getelementptr inbounds nuw %XLA_CPU_KernelThreadDim, ptr %tdims, i32 0, i32 1
  %tdim_z_gep = getelementptr inbounds nuw %XLA_CPU_KernelThreadDim, ptr %tdims, i32 0, i32 2
  %tdim_x = load i64, ptr %tdim_x_gep, align 4
  %tdim_y = load i64, ptr %tdim_y_gep, align 4
  %tdim_z = load i64, ptr %tdim_z_gep, align 4
  %tid_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 1
  %tids = load ptr, ptr %tid_gep, align 8
  %tid_x_gep = getelementptr inbounds nuw %XLA_CPU_KernelThread, ptr %tids, i32 0, i32 0
  %tid_y_gep = getelementptr inbounds nuw %XLA_CPU_KernelThread, ptr %tids, i32 0, i32 1
  %tid_z_gep = getelementptr inbounds nuw %XLA_CPU_KernelThread, ptr %tids, i32 0, i32 2
  %tid_x = load i64, ptr %tid_x_gep, align 4
  %tid_y = load i64, ptr %tid_y_gep, align 4
  %tid_z = load i64, ptr %tid_z_gep, align 4
  %args_gep = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args = load ptr, ptr %args_gep, align 8
  %arg0_gep = getelementptr %XLA_CPU_KernelArg, ptr %args, i32 0, i32 0
  %arg0 = load ptr, ptr %arg0_gep, align 8, !invariant.load !1, !dereferenceable !2, !align !3
  %args_gep1 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args2 = load ptr, ptr %args_gep1, align 8
  %arg1_gep = getelementptr %XLA_CPU_KernelArg, ptr %args2, i32 1, i32 0
  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !1, !dereferenceable !2, !align !3
  %args_gep3 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args4 = load ptr, ptr %args_gep3, align 8
  %arg2_gep = getelementptr %XLA_CPU_KernelArg, ptr %args4, i32 2, i32 0
  %arg2 = load ptr, ptr %arg2_gep, align 8, !invariant.load !1, !dereferenceable !2, !align !3
  %lo_dim_0_gep = getelementptr inbounds [18 x [2 x [2 x i64]]], ptr @broadcast_multiply_fusion.clone_parallel_bounds, i32 0, i64 %tid_x, i32 0, i32 0
  %up_dim_0_gep = getelementptr inbounds [18 x [2 x [2 x i64]]], ptr @broadcast_multiply_fusion.clone_parallel_bounds, i32 0, i64 %tid_x, i32 0, i32 1
  %lo_dim_0 = load i64, ptr %lo_dim_0_gep, align 4
  %up_dim_0 = load i64, ptr %up_dim_0_gep, align 4
  %lo_dim_1_gep = getelementptr inbounds [18 x [2 x [2 x i64]]], ptr @broadcast_multiply_fusion.clone_parallel_bounds, i32 0, i64 %tid_x, i32 1, i32 0
  %up_dim_1_gep = getelementptr inbounds [18 x [2 x [2 x i64]]], ptr @broadcast_multiply_fusion.clone_parallel_bounds, i32 0, i64 %tid_x, i32 1, i32 1
  %lo_dim_1 = load i64, ptr %lo_dim_1_gep, align 4
  %up_dim_1 = load i64, ptr %up_dim_1_gep, align 4
  store i64 %lo_dim_0, ptr %broadcast_multiply_fusion.clone.invar_address.dim.0, align 4
  br label %broadcast_multiply_fusion.clone.loop_header.dim.0

broadcast_multiply_fusion.clone.loop_header.dim.0: ; preds = %broadcast_multiply_fusion.clone.loop_exit.dim.1, %1
  %broadcast_multiply_fusion.clone.indvar.dim.0 = load i64, ptr %broadcast_multiply_fusion.clone.invar_address.dim.0, align 4
  %2 = icmp uge i64 %broadcast_multiply_fusion.clone.indvar.dim.0, %up_dim_0
  br i1 %2, label %broadcast_multiply_fusion.clone.loop_exit.dim.0, label %broadcast_multiply_fusion.clone.loop_body.dim.0

broadcast_multiply_fusion.clone.loop_body.dim.0:  ; preds = %broadcast_multiply_fusion.clone.loop_header.dim.0
  store i64 %lo_dim_1, ptr %broadcast_multiply_fusion.clone.invar_address.dim.1, align 4
  br label %broadcast_multiply_fusion.clone.loop_header.dim.1

broadcast_multiply_fusion.clone.loop_header.dim.1: ; preds = %broadcast_multiply_fusion.clone.loop_exit.dim.2, %broadcast_multiply_fusion.clone.loop_body.dim.0
  %broadcast_multiply_fusion.clone.indvar.dim.1 = load i64, ptr %broadcast_multiply_fusion.clone.invar_address.dim.1, align 4
  %3 = icmp uge i64 %broadcast_multiply_fusion.clone.indvar.dim.1, %up_dim_1
  br i1 %3, label %broadcast_multiply_fusion.clone.loop_exit.dim.1, label %broadcast_multiply_fusion.clone.loop_body.dim.1

broadcast_multiply_fusion.clone.loop_body.dim.1:  ; preds = %broadcast_multiply_fusion.clone.loop_header.dim.1
  store i64 0, ptr %broadcast_multiply_fusion.clone.invar_address.dim.2, align 4
  br label %broadcast_multiply_fusion.clone.loop_header.dim.2

broadcast_multiply_fusion.clone.loop_header.dim.2: ; preds = %broadcast_multiply_fusion.clone.loop_body.dim.2, %broadcast_multiply_fusion.clone.loop_body.dim.1
  %broadcast_multiply_fusion.clone.indvar.dim.2 = load i64, ptr %broadcast_multiply_fusion.clone.invar_address.dim.2, align 4
  %4 = icmp uge i64 %broadcast_multiply_fusion.clone.indvar.dim.2, 64
  br i1 %4, label %broadcast_multiply_fusion.clone.loop_exit.dim.2, label %broadcast_multiply_fusion.clone.loop_body.dim.2

broadcast_multiply_fusion.clone.loop_body.dim.2:  ; preds = %broadcast_multiply_fusion.clone.loop_header.dim.2
  %constant.151 = load float, ptr @__llvmsplit_unnamed.15, align 4
  %5 = getelementptr inbounds [3 x [10000 x [64 x i32]]], ptr %arg0, i64 0, i64 %broadcast_multiply_fusion.clone.indvar.dim.0, i64 %broadcast_multiply_fusion.clone.indvar.dim.1, i64 %broadcast_multiply_fusion.clone.indvar.dim.2
  %6 = load i32, ptr %5, align 4, !invariant.load !1, !noalias !4
  %7 = getelementptr inbounds [3 x [10000 x [64 x i32]]], ptr %arg1, i64 0, i64 %broadcast_multiply_fusion.clone.indvar.dim.0, i64 %broadcast_multiply_fusion.clone.indvar.dim.1, i64 %broadcast_multiply_fusion.clone.indvar.dim.2
  %8 = load i32, ptr %7, align 4, !invariant.load !1, !noalias !4
  %9 = xor i32 %6, %8
  %constant.150 = load i32, ptr @__llvmsplit_unnamed.19, align 4
  %10 = lshr i32 %9, %constant.150
  %shft.chk = icmp ult i32 %constant.150, 32
  %11 = select i1 %shft.chk, i32 %10, i32 0
  %constant.149 = load i32, ptr @__llvmsplit_unnamed.18, align 4
  %12 = or i32 %11, %constant.149
  %13 = bitcast i32 %12 to float
  %constant.148 = load float, ptr @__llvmsplit_unnamed.17, align 4
  %add.126 = fadd float %13, %constant.148
  %constant.147 = load float, ptr @__llvmsplit_unnamed.16, align 4
  %multiply.53 = fmul float %add.126, %constant.147
  %add.125 = fadd float %multiply.53, %constant.151
  %14 = call float @llvm.maximum.f32(float %constant.151, float %add.125)
  %15 = call float @llvm.fabs.f32(float %14)
  %constant.146 = load float, ptr @__llvmsplit_unnamed.42, align 4
  %compare.7 = fcmp oeq float %15, %constant.146
  %16 = zext i1 %compare.7 to i8
  %constant.145 = load float, ptr @__llvmsplit_unnamed.41, align 4
  %multiply.52 = fmul float %14, %constant.145
  %17 = fneg float %14
  %multiply.51 = fmul float %14, %17
  %18 = fadd float %multiply.51, 1.000000e+00
  %19 = call float @llvm.log.f32(float %18)
  %20 = fmul float %multiply.51, %multiply.51
  %21 = fmul float 0.000000e+00, %multiply.51
  %22 = fadd float %21, 1.000000e+00
  %23 = fmul float %22, %multiply.51
  %24 = fadd float %23, 0x402E2035A0000000
  %25 = fmul float %24, %multiply.51
  %26 = fadd float %25, 0x4054C30B60000000
  %27 = fmul float %26, %multiply.51
  %28 = fadd float %27, 0x406BB865A0000000
  %29 = fmul float %28, %multiply.51
  %30 = fadd float %29, 0x4073519460000000
  %31 = fmul float %30, %multiply.51
  %32 = fadd float %31, 0x406B0DB140000000
  %33 = fmul float %32, %multiply.51
  %34 = fadd float %33, 0x404E0F3040000000
  %35 = fmul float 0.000000e+00, %multiply.51
  %36 = fadd float %35, 0x3F07BC0960000000
  %37 = fmul float %36, %multiply.51
  %38 = fadd float %37, 0x3FDFE818A0000000
  %39 = fmul float %38, %multiply.51
  %40 = fadd float %39, 0x401A509F40000000
  %41 = fmul float %40, %multiply.51
  %42 = fadd float %41, 0x403DE97380000000
  %43 = fmul float %42, %multiply.51
  %44 = fadd float %43, 0x404E798EC0000000
  %45 = fmul float %44, %multiply.51
  %46 = fadd float %45, 0x404C8E75A0000000
  %47 = fmul float %46, %multiply.51
  %48 = fadd float %47, 0x40340A2020000000
  %49 = fdiv float %48, %34
  %50 = fmul float %multiply.51, %20
  %51 = fmul float %50, %49
  %52 = fmul float -5.000000e-01, %20
  %53 = fadd float %52, %51
  %54 = fadd float %multiply.51, %53
  %55 = call float @llvm.fabs.f32(float %multiply.51)
  %56 = fcmp olt float %55, 0x3FDA8279A0000000
  %57 = select i1 %56, float %54, float %19
  %58 = fneg float %57
  %constant.144 = load float, ptr @__llvmsplit_unnamed.22, align 4
  %compare.6 = fcmp olt float %58, %constant.144
  %59 = zext i1 %compare.6 to i8
  %constant.143 = load float, ptr @__llvmsplit_unnamed.40, align 4
  %constant.142 = load float, ptr @__llvmsplit_unnamed.39, align 4
  %60 = trunc i8 %59 to i1
  %61 = select i1 %60, float %constant.143, float %constant.142
  %constant.141 = load float, ptr @__llvmsplit_unnamed.38, align 4
  %constant.140 = load float, ptr @__llvmsplit_unnamed.37, align 4
  %62 = trunc i8 %59 to i1
  %63 = select i1 %62, float %constant.141, float %constant.140
  %constant.139 = load float, ptr @__llvmsplit_unnamed.36, align 4
  %constant.138 = load float, ptr @__llvmsplit_unnamed.35, align 4
  %64 = trunc i8 %59 to i1
  %65 = select i1 %64, float %constant.139, float %constant.138
  %constant.137 = load float, ptr @__llvmsplit_unnamed.34, align 4
  %constant.136 = load float, ptr @__llvmsplit_unnamed.33, align 4
  %66 = trunc i8 %59 to i1
  %67 = select i1 %66, float %constant.137, float %constant.136
  %constant.135 = load float, ptr @__llvmsplit_unnamed.32, align 4
  %constant.134 = load float, ptr @__llvmsplit_unnamed.31, align 4
  %68 = trunc i8 %59 to i1
  %69 = select i1 %68, float %constant.135, float %constant.134
  %constant.133 = load float, ptr @__llvmsplit_unnamed.30, align 4
  %constant.132 = load float, ptr @__llvmsplit_unnamed.29, align 4
  %70 = trunc i8 %59 to i1
  %71 = select i1 %70, float %constant.133, float %constant.132
  %constant.131 = load float, ptr @__llvmsplit_unnamed.28, align 4
  %constant.130 = load float, ptr @__llvmsplit_unnamed.27, align 4
  %72 = trunc i8 %59 to i1
  %73 = select i1 %72, float %constant.131, float %constant.130
  %constant.129 = load float, ptr @__llvmsplit_unnamed.26, align 4
  %constant.128 = load float, ptr @__llvmsplit_unnamed.25, align 4
  %74 = trunc i8 %59 to i1
  %75 = select i1 %74, float %constant.129, float %constant.128
  %constant.127 = load float, ptr @__llvmsplit_unnamed.24, align 4
  %constant.126 = load float, ptr @__llvmsplit_unnamed.23, align 4
  %76 = trunc i8 %59 to i1
  %77 = select i1 %76, float %constant.127, float %constant.126
  %constant.125 = load float, ptr @__llvmsplit_unnamed.21, align 4
  %add.124 = fadd float %58, %constant.125
  %78 = call float @llvm.sqrt.f32(float %58)
  %constant.124 = load float, ptr @__llvmsplit_unnamed.20, align 4
  %add.123 = fadd float %78, %constant.124
  %79 = trunc i8 %59 to i1
  %80 = select i1 %79, float %add.124, float %add.123
  %multiply.50 = fmul float %77, %80
  %add.122 = fadd float %75, %multiply.50
  %multiply.49 = fmul float %add.122, %80
  %add.121 = fadd float %73, %multiply.49
  %multiply.48 = fmul float %add.121, %80
  %add.120 = fadd float %71, %multiply.48
  %multiply.47 = fmul float %add.120, %80
  %add.119 = fadd float %69, %multiply.47
  %multiply.46 = fmul float %add.119, %80
  %add.118 = fadd float %67, %multiply.46
  %multiply.45 = fmul float %add.118, %80
  %add.117 = fadd float %65, %multiply.45
  %multiply.44 = fmul float %add.117, %80
  %add.116 = fadd float %63, %multiply.44
  %multiply.43 = fmul float %add.116, %80
  %add.115 = fadd float %61, %multiply.43
  %multiply.42 = fmul float %add.115, %14
  %81 = trunc i8 %16 to i1
  %82 = select i1 %81, float %multiply.52, float %multiply.42
  %constant.123 = load float, ptr @__llvmsplit_unnamed.14, align 4
  %multiply.41 = fmul float %82, %constant.123
  %83 = getelementptr inbounds [3 x [10000 x [64 x float]]], ptr %arg2, i64 0, i64 %broadcast_multiply_fusion.clone.indvar.dim.0, i64 %broadcast_multiply_fusion.clone.indvar.dim.1, i64 %broadcast_multiply_fusion.clone.indvar.dim.2
  store float %multiply.41, ptr %83, align 4, !alias.scope !4
  %invar.inc6 = add nuw nsw i64 %broadcast_multiply_fusion.clone.indvar.dim.2, 1
  store i64 %invar.inc6, ptr %broadcast_multiply_fusion.clone.invar_address.dim.2, align 4
  br label %broadcast_multiply_fusion.clone.loop_header.dim.2

broadcast_multiply_fusion.clone.loop_exit.dim.2:  ; preds = %broadcast_multiply_fusion.clone.loop_header.dim.2
  %invar.inc5 = add nuw nsw i64 %broadcast_multiply_fusion.clone.indvar.dim.1, 1
  store i64 %invar.inc5, ptr %broadcast_multiply_fusion.clone.invar_address.dim.1, align 4
  br label %broadcast_multiply_fusion.clone.loop_header.dim.1, !llvm.loop !7

broadcast_multiply_fusion.clone.loop_exit.dim.1:  ; preds = %broadcast_multiply_fusion.clone.loop_header.dim.1
  %invar.inc = add nuw nsw i64 %broadcast_multiply_fusion.clone.indvar.dim.0, 1
  store i64 %invar.inc, ptr %broadcast_multiply_fusion.clone.invar_address.dim.0, align 4
  br label %broadcast_multiply_fusion.clone.loop_header.dim.0, !llvm.loop !9

broadcast_multiply_fusion.clone.loop_exit.dim.0:  ; preds = %broadcast_multiply_fusion.clone.loop_header.dim.0
  br label %return

return:                                           ; preds = %broadcast_multiply_fusion.clone.loop_exit.dim.0
  ret ptr null
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maximum.f32(float, float) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fabs.f32(float) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.log.f32(float) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sqrt.f32(float) #1

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"xla_dylib_index", i64 0}
!1 = !{}
!2 = !{i64 7680000}
!3 = !{i64 64}
!4 = !{!5}
!5 = !{!"result slice: {index:0, offset:0, size:7680000}", !6}
!6 = !{!"XLA host kernel broadcast_multiply_fusion.clone AA domain"}
!7 = distinct !{!7, !8}
!8 = !{!"llvm.loop.unroll.disable"}
!9 = distinct !{!9, !8}
