; ModuleID = '__compute_module_part_01'
source_filename = "__compute_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_KernelThreadDim = type { i64, i64, i64 }
%XLA_CPU_KernelThread = type { i64, i64, i64 }
%XLA_CPU_KernelArg = type { ptr, i64 }

@__llvmsplit_unnamed.8 = private unnamed_addr constant [8 x i8] c" \00\00\00\00\00\00\00"
@__llvmsplit_unnamed.9 = private unnamed_addr constant [8 x i8] c"@\00\00\00\00\00\00\00"
@__llvmsplit_unnamed.10 = private unnamed_addr constant [8 x i8] c"\00\C4\09\00\00\00\00\00"
@broadcast_add_fusion.3.clone_parallel_bounds = private constant [18 x [2 x [2 x i64]]] [[2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 0, i64 1666]], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 1666, i64 3332]], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 3332, i64 4998]], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 4998, i64 6664]], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 6664, i64 8330]], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 8330, i64 10000]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 2], [2 x i64] [i64 0, i64 1666]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 2], [2 x i64] [i64 1666, i64 3332]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 2], [2 x i64] [i64 3332, i64 4998]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 2], [2 x i64] [i64 4998, i64 6664]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 2], [2 x i64] [i64 6664, i64 8330]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 2], [2 x i64] [i64 8330, i64 10000]], [2 x [2 x i64]] [[2 x i64] [i64 2, i64 3], [2 x i64] [i64 0, i64 1666]], [2 x [2 x i64]] [[2 x i64] [i64 2, i64 3], [2 x i64] [i64 1666, i64 3332]], [2 x [2 x i64]] [[2 x i64] [i64 2, i64 3], [2 x i64] [i64 3332, i64 4998]], [2 x [2 x i64]] [[2 x i64] [i64 2, i64 3], [2 x i64] [i64 4998, i64 6664]], [2 x [2 x i64]] [[2 x i64] [i64 2, i64 3], [2 x i64] [i64 6664, i64 8330]], [2 x [2 x i64]] [[2 x i64] [i64 2, i64 3], [2 x i64] [i64 8330, i64 10000]]]

; Function Attrs: uwtable
define ptr @broadcast_add_fusion.3.clone(ptr %0) #0 {
  %broadcast_add_fusion.3.clone.invar_address.dim.2 = alloca i64, align 8
  %broadcast_add_fusion.3.clone.invar_address.dim.1 = alloca i64, align 8
  %broadcast_add_fusion.3.clone.invar_address.dim.0 = alloca i64, align 8
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
  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !1, !dereferenceable !4, !align !3
  %lo_dim_0_gep = getelementptr inbounds [18 x [2 x [2 x i64]]], ptr @broadcast_add_fusion.3.clone_parallel_bounds, i32 0, i64 %tid_x, i32 0, i32 0
  %up_dim_0_gep = getelementptr inbounds [18 x [2 x [2 x i64]]], ptr @broadcast_add_fusion.3.clone_parallel_bounds, i32 0, i64 %tid_x, i32 0, i32 1
  %lo_dim_0 = load i64, ptr %lo_dim_0_gep, align 4
  %up_dim_0 = load i64, ptr %up_dim_0_gep, align 4
  %lo_dim_1_gep = getelementptr inbounds [18 x [2 x [2 x i64]]], ptr @broadcast_add_fusion.3.clone_parallel_bounds, i32 0, i64 %tid_x, i32 1, i32 0
  %up_dim_1_gep = getelementptr inbounds [18 x [2 x [2 x i64]]], ptr @broadcast_add_fusion.3.clone_parallel_bounds, i32 0, i64 %tid_x, i32 1, i32 1
  %lo_dim_1 = load i64, ptr %lo_dim_1_gep, align 4
  %up_dim_1 = load i64, ptr %up_dim_1_gep, align 4
  store i64 %lo_dim_0, ptr %broadcast_add_fusion.3.clone.invar_address.dim.0, align 4
  br label %broadcast_add_fusion.3.clone.loop_header.dim.0

broadcast_add_fusion.3.clone.loop_header.dim.0:   ; preds = %broadcast_add_fusion.3.clone.loop_exit.dim.1, %1
  %broadcast_add_fusion.3.clone.indvar.dim.0 = load i64, ptr %broadcast_add_fusion.3.clone.invar_address.dim.0, align 4
  %2 = icmp uge i64 %broadcast_add_fusion.3.clone.indvar.dim.0, %up_dim_0
  br i1 %2, label %broadcast_add_fusion.3.clone.loop_exit.dim.0, label %broadcast_add_fusion.3.clone.loop_body.dim.0

broadcast_add_fusion.3.clone.loop_body.dim.0:     ; preds = %broadcast_add_fusion.3.clone.loop_header.dim.0
  store i64 %lo_dim_1, ptr %broadcast_add_fusion.3.clone.invar_address.dim.1, align 4
  br label %broadcast_add_fusion.3.clone.loop_header.dim.1

broadcast_add_fusion.3.clone.loop_header.dim.1:   ; preds = %broadcast_add_fusion.3.clone.loop_exit.dim.2, %broadcast_add_fusion.3.clone.loop_body.dim.0
  %broadcast_add_fusion.3.clone.indvar.dim.1 = load i64, ptr %broadcast_add_fusion.3.clone.invar_address.dim.1, align 4
  %3 = icmp uge i64 %broadcast_add_fusion.3.clone.indvar.dim.1, %up_dim_1
  br i1 %3, label %broadcast_add_fusion.3.clone.loop_exit.dim.1, label %broadcast_add_fusion.3.clone.loop_body.dim.1

broadcast_add_fusion.3.clone.loop_body.dim.1:     ; preds = %broadcast_add_fusion.3.clone.loop_header.dim.1
  store i64 0, ptr %broadcast_add_fusion.3.clone.invar_address.dim.2, align 4
  br label %broadcast_add_fusion.3.clone.loop_header.dim.2

broadcast_add_fusion.3.clone.loop_header.dim.2:   ; preds = %broadcast_add_fusion.3.clone.loop_body.dim.2, %broadcast_add_fusion.3.clone.loop_body.dim.1
  %broadcast_add_fusion.3.clone.indvar.dim.2 = load i64, ptr %broadcast_add_fusion.3.clone.invar_address.dim.2, align 4
  %4 = icmp uge i64 %broadcast_add_fusion.3.clone.indvar.dim.2, 64
  br i1 %4, label %broadcast_add_fusion.3.clone.loop_exit.dim.2, label %broadcast_add_fusion.3.clone.loop_body.dim.2

broadcast_add_fusion.3.clone.loop_body.dim.2:     ; preds = %broadcast_add_fusion.3.clone.loop_header.dim.2
  %5 = mul nuw nsw i64 %broadcast_add_fusion.3.clone.indvar.dim.0, 1
  %6 = add nuw nsw i64 0, %5
  %constant.156 = load i64, ptr @__llvmsplit_unnamed.10, align 4
  %7 = mul i64 %6, %constant.156
  %8 = mul nuw nsw i64 %broadcast_add_fusion.3.clone.indvar.dim.1, 1
  %9 = add nuw nsw i64 0, %8
  %constant.155 = load i64, ptr @__llvmsplit_unnamed.9, align 4
  %10 = mul i64 %9, %constant.155
  %11 = add i64 %7, %10
  %12 = mul nuw nsw i64 %broadcast_add_fusion.3.clone.indvar.dim.2, 1
  %13 = add nuw nsw i64 0, %12
  %14 = add i64 %11, %13
  %constant.154 = load i64, ptr @__llvmsplit_unnamed.8, align 4
  %15 = lshr i64 %14, %constant.154
  %shft.chk = icmp ult i64 %constant.154, 64
  %16 = select i1 %shft.chk, i64 %15, i64 0
  %17 = trunc i64 %16 to i32
  %18 = getelementptr inbounds [2 x i32], ptr %arg0, i64 0, i64 0
  %19 = load i32, ptr %18, align 4, !invariant.load !1, !noalias !5
  %20 = add i32 %17, %19
  %21 = getelementptr inbounds [3 x [10000 x [64 x i32]]], ptr %arg1, i64 0, i64 %broadcast_add_fusion.3.clone.indvar.dim.0, i64 %broadcast_add_fusion.3.clone.indvar.dim.1, i64 %broadcast_add_fusion.3.clone.indvar.dim.2
  store i32 %20, ptr %21, align 4, !alias.scope !5
  %invar.inc4 = add nuw nsw i64 %broadcast_add_fusion.3.clone.indvar.dim.2, 1
  store i64 %invar.inc4, ptr %broadcast_add_fusion.3.clone.invar_address.dim.2, align 4
  br label %broadcast_add_fusion.3.clone.loop_header.dim.2

broadcast_add_fusion.3.clone.loop_exit.dim.2:     ; preds = %broadcast_add_fusion.3.clone.loop_header.dim.2
  %invar.inc3 = add nuw nsw i64 %broadcast_add_fusion.3.clone.indvar.dim.1, 1
  store i64 %invar.inc3, ptr %broadcast_add_fusion.3.clone.invar_address.dim.1, align 4
  br label %broadcast_add_fusion.3.clone.loop_header.dim.1, !llvm.loop !8

broadcast_add_fusion.3.clone.loop_exit.dim.1:     ; preds = %broadcast_add_fusion.3.clone.loop_header.dim.1
  %invar.inc = add nuw nsw i64 %broadcast_add_fusion.3.clone.indvar.dim.0, 1
  store i64 %invar.inc, ptr %broadcast_add_fusion.3.clone.invar_address.dim.0, align 4
  br label %broadcast_add_fusion.3.clone.loop_header.dim.0, !llvm.loop !10

broadcast_add_fusion.3.clone.loop_exit.dim.0:     ; preds = %broadcast_add_fusion.3.clone.loop_header.dim.0
  br label %return

return:                                           ; preds = %broadcast_add_fusion.3.clone.loop_exit.dim.0
  ret ptr null
}

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"xla_dylib_index", i64 1}
!1 = !{}
!2 = !{i64 8}
!3 = !{i64 64}
!4 = !{i64 7680000}
!5 = !{!6}
!6 = !{!"result slice: {index:7, offset:15360064, size:7680000}", !7}
!7 = !{!"XLA host kernel broadcast_add_fusion.3.clone AA domain"}
!8 = distinct !{!8, !9}
!9 = !{!"llvm.loop.unroll.disable"}
!10 = distinct !{!10, !9}
