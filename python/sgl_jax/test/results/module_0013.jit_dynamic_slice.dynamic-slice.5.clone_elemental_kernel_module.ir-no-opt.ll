; ModuleID = '__compute_module_dynamic-slice.5.clone_elemental_kernel_module'
source_filename = "__compute_module_dynamic-slice.5.clone_elemental_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_KernelThreadDim = type { i64, i64, i64 }
%XLA_CPU_KernelThread = type { i64, i64, i64 }
%XLA_CPU_KernelArg = type { ptr, i64 }

@0 = private unnamed_addr constant [4 x i8] zeroinitializer, align 4
@dynamic-slice.5.clone_parallel_bounds = private constant [4 x [2 x [2 x i64]]] [[2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 0, i64 2500]], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 2500, i64 5000]], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 5000, i64 7500]], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 7500, i64 10000]]]

; Function Attrs: uwtable
define ptr @dynamic-slice.5.clone_kernel(ptr %0) #0 {
  %dynamic-slice.5.clone.invar_address.dim.2 = alloca i64, align 8
  %dynamic-slice.5.clone.invar_address.dim.1 = alloca i64, align 8
  %dynamic-slice.5.clone.invar_address.dim.0 = alloca i64, align 8
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
  %args_gep3 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args4 = load ptr, ptr %args_gep3, align 8
  %arg2_gep = getelementptr %XLA_CPU_KernelArg, ptr %args4, i32 2, i32 0
  %arg2 = load ptr, ptr %arg2_gep, align 8, !invariant.load !1, !dereferenceable !4, !align !3
  %args_gep5 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args6 = load ptr, ptr %args_gep5, align 8
  %arg3_gep = getelementptr %XLA_CPU_KernelArg, ptr %args6, i32 3, i32 0
  %arg3 = load ptr, ptr %arg3_gep, align 8, !invariant.load !1, !dereferenceable !4, !align !3
  %args_gep7 = getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, ptr %0, i32 0, i32 3
  %args8 = load ptr, ptr %args_gep7, align 8
  %arg4_gep = getelementptr %XLA_CPU_KernelArg, ptr %args8, i32 4, i32 0
  %arg4 = load ptr, ptr %arg4_gep, align 8, !invariant.load !1, !dereferenceable !5, !align !3
  %lo_dim_0_gep = getelementptr inbounds [4 x [2 x [2 x i64]]], ptr @dynamic-slice.5.clone_parallel_bounds, i32 0, i64 %tid_x, i32 0, i32 0
  %up_dim_0_gep = getelementptr inbounds [4 x [2 x [2 x i64]]], ptr @dynamic-slice.5.clone_parallel_bounds, i32 0, i64 %tid_x, i32 0, i32 1
  %lo_dim_0 = load i64, ptr %lo_dim_0_gep, align 4
  %up_dim_0 = load i64, ptr %up_dim_0_gep, align 4
  %lo_dim_1_gep = getelementptr inbounds [4 x [2 x [2 x i64]]], ptr @dynamic-slice.5.clone_parallel_bounds, i32 0, i64 %tid_x, i32 1, i32 0
  %up_dim_1_gep = getelementptr inbounds [4 x [2 x [2 x i64]]], ptr @dynamic-slice.5.clone_parallel_bounds, i32 0, i64 %tid_x, i32 1, i32 1
  %lo_dim_1 = load i64, ptr %lo_dim_1_gep, align 4
  %up_dim_1 = load i64, ptr %up_dim_1_gep, align 4
  store i64 %lo_dim_0, ptr %dynamic-slice.5.clone.invar_address.dim.0, align 4
  br label %dynamic-slice.5.clone.loop_header.dim.0

dynamic-slice.5.clone.loop_header.dim.0:          ; preds = %dynamic-slice.5.clone.loop_exit.dim.1, %1
  %dynamic-slice.5.clone.indvar.dim.0 = load i64, ptr %dynamic-slice.5.clone.invar_address.dim.0, align 4
  %2 = icmp uge i64 %dynamic-slice.5.clone.indvar.dim.0, %up_dim_0
  br i1 %2, label %dynamic-slice.5.clone.loop_exit.dim.0, label %dynamic-slice.5.clone.loop_body.dim.0

dynamic-slice.5.clone.loop_body.dim.0:            ; preds = %dynamic-slice.5.clone.loop_header.dim.0
  store i64 %lo_dim_1, ptr %dynamic-slice.5.clone.invar_address.dim.1, align 4
  br label %dynamic-slice.5.clone.loop_header.dim.1

dynamic-slice.5.clone.loop_header.dim.1:          ; preds = %dynamic-slice.5.clone.loop_exit.dim.2, %dynamic-slice.5.clone.loop_body.dim.0
  %dynamic-slice.5.clone.indvar.dim.1 = load i64, ptr %dynamic-slice.5.clone.invar_address.dim.1, align 4
  %3 = icmp uge i64 %dynamic-slice.5.clone.indvar.dim.1, %up_dim_1
  br i1 %3, label %dynamic-slice.5.clone.loop_exit.dim.1, label %dynamic-slice.5.clone.loop_body.dim.1

dynamic-slice.5.clone.loop_body.dim.1:            ; preds = %dynamic-slice.5.clone.loop_header.dim.1
  store i64 0, ptr %dynamic-slice.5.clone.invar_address.dim.2, align 4
  br label %dynamic-slice.5.clone.loop_header.dim.2

dynamic-slice.5.clone.loop_header.dim.2:          ; preds = %dynamic-slice.5.clone.loop_body.dim.2, %dynamic-slice.5.clone.loop_body.dim.1
  %dynamic-slice.5.clone.indvar.dim.2 = load i64, ptr %dynamic-slice.5.clone.invar_address.dim.2, align 4
  %4 = icmp uge i64 %dynamic-slice.5.clone.indvar.dim.2, 16
  br i1 %4, label %dynamic-slice.5.clone.loop_exit.dim.2, label %dynamic-slice.5.clone.loop_body.dim.2

dynamic-slice.5.clone.loop_body.dim.2:            ; preds = %dynamic-slice.5.clone.loop_header.dim.2
  %5 = load i32, ptr %arg1, align 4, !invariant.load !1, !noalias !6
  %6 = sext i32 %5 to i64
  %7 = icmp sge i64 0, %6
  %8 = select i1 %7, i64 0, i64 %6
  %9 = icmp sle i64 2, %8
  %dynamic-slice.5.clone.start_idx0 = select i1 %9, i64 2, i64 %8
  %10 = load i32, ptr %arg3, align 4, !invariant.load !1, !noalias !6
  %11 = sext i32 %10 to i64
  %12 = icmp sge i64 0, %11
  %13 = select i1 %12, i64 0, i64 %11
  %14 = icmp sle i64 0, %13
  %dynamic-slice.5.clone.start_idx1 = select i1 %14, i64 0, i64 %13
  %15 = load i32, ptr %arg3, align 4, !invariant.load !1, !noalias !6
  %16 = sext i32 %15 to i64
  %17 = icmp sge i64 0, %16
  %18 = select i1 %17, i64 0, i64 %16
  %19 = icmp sle i64 0, %18
  %dynamic-slice.5.clone.start_idx2 = select i1 %19, i64 0, i64 %18
  %20 = add i64 %dynamic-slice.5.clone.start_idx0, %dynamic-slice.5.clone.indvar.dim.0
  %21 = add i64 %dynamic-slice.5.clone.start_idx1, %dynamic-slice.5.clone.indvar.dim.1
  %22 = add i64 %dynamic-slice.5.clone.start_idx2, %dynamic-slice.5.clone.indvar.dim.2
  %23 = getelementptr inbounds [3 x [10000 x [16 x float]]], ptr %arg0, i64 0, i64 %20, i64 %21, i64 %22
  %24 = load float, ptr %23, align 4, !invariant.load !1, !noalias !6
  %25 = getelementptr inbounds [1 x [10000 x [16 x float]]], ptr %arg4, i64 0, i64 0, i64 %dynamic-slice.5.clone.indvar.dim.1, i64 %dynamic-slice.5.clone.indvar.dim.2
  store float %24, ptr %25, align 4, !alias.scope !6
  %invar.inc10 = add nuw nsw i64 %dynamic-slice.5.clone.indvar.dim.2, 1
  store i64 %invar.inc10, ptr %dynamic-slice.5.clone.invar_address.dim.2, align 4
  br label %dynamic-slice.5.clone.loop_header.dim.2

dynamic-slice.5.clone.loop_exit.dim.2:            ; preds = %dynamic-slice.5.clone.loop_header.dim.2
  %invar.inc9 = add nuw nsw i64 %dynamic-slice.5.clone.indvar.dim.1, 1
  store i64 %invar.inc9, ptr %dynamic-slice.5.clone.invar_address.dim.1, align 4
  br label %dynamic-slice.5.clone.loop_header.dim.1, !llvm.loop !9

dynamic-slice.5.clone.loop_exit.dim.1:            ; preds = %dynamic-slice.5.clone.loop_header.dim.1
  %invar.inc = add nuw nsw i64 %dynamic-slice.5.clone.indvar.dim.0, 1
  store i64 %invar.inc, ptr %dynamic-slice.5.clone.invar_address.dim.0, align 4
  br label %dynamic-slice.5.clone.loop_header.dim.0, !llvm.loop !11

dynamic-slice.5.clone.loop_exit.dim.0:            ; preds = %dynamic-slice.5.clone.loop_header.dim.0
  br label %return

return:                                           ; preds = %dynamic-slice.5.clone.loop_exit.dim.0
  ret ptr null
}

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"xla_dylib_index", i64 0}
!1 = !{}
!2 = !{i64 1920000}
!3 = !{i64 64}
!4 = !{i64 4}
!5 = !{i64 640000}
!6 = !{!7}
!7 = !{!"result slice: {index:1, offset:0, size:640000}", !8}
!8 = !{!"XLA host kernel dynamic-slice.5.clone_kernel AA domain"}
!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.unroll.disable"}
!11 = distinct !{!11, !10}
