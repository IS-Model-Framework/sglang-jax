; ModuleID = '__compute_module_part_01'
source_filename = "__compute_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%XLA_CPU_KernelCallFrame = type { ptr, ptr, i64, ptr }
%XLA_CPU_KernelThreadDim = type { i64, i64, i64 }
%XLA_CPU_KernelThread = type { i64, i64, i64 }
%XLA_CPU_KernelArg = type { ptr, i64 }

@bitcast_concatenate_fusion.1.clone_parallel_bounds = private constant [5 x [1 x [2 x i64]]] [[1 x [2 x i64]] [[2 x i64] [i64 0, i64 2000]], [1 x [2 x i64]] [[2 x i64] [i64 2000, i64 4000]], [1 x [2 x i64]] [[2 x i64] [i64 4000, i64 6000]], [1 x [2 x i64]] [[2 x i64] [i64 6000, i64 8000]], [1 x [2 x i64]] [[2 x i64] [i64 8000, i64 10000]]]

; Function Attrs: uwtable
define ptr @bitcast_concatenate_fusion.1.clone(ptr %0) #0 {
  %bitcast_concatenate_fusion.1.clone.invar_address.dim.1 = alloca i64, align 8
  %bitcast_concatenate_fusion.1.clone.invar_address.dim.0 = alloca i64, align 8
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
  %lo_dim_0_gep = getelementptr inbounds [5 x [1 x [2 x i64]]], ptr @bitcast_concatenate_fusion.1.clone_parallel_bounds, i32 0, i64 %tid_x, i32 0, i32 0
  %up_dim_0_gep = getelementptr inbounds [5 x [1 x [2 x i64]]], ptr @bitcast_concatenate_fusion.1.clone_parallel_bounds, i32 0, i64 %tid_x, i32 0, i32 1
  %lo_dim_0 = load i64, ptr %lo_dim_0_gep, align 4
  %up_dim_0 = load i64, ptr %up_dim_0_gep, align 4
  store i64 %lo_dim_0, ptr %bitcast_concatenate_fusion.1.clone.invar_address.dim.0, align 4
  br label %bitcast_concatenate_fusion.1.clone.loop_header.dim.0

bitcast_concatenate_fusion.1.clone.loop_header.dim.0: ; preds = %bitcast_concatenate_fusion.1.clone.loop_exit.dim.1, %1
  %bitcast_concatenate_fusion.1.clone.indvar.dim.0 = load i64, ptr %bitcast_concatenate_fusion.1.clone.invar_address.dim.0, align 4
  %2 = icmp uge i64 %bitcast_concatenate_fusion.1.clone.indvar.dim.0, %up_dim_0
  br i1 %2, label %bitcast_concatenate_fusion.1.clone.loop_exit.dim.0, label %bitcast_concatenate_fusion.1.clone.loop_body.dim.0

bitcast_concatenate_fusion.1.clone.loop_body.dim.0: ; preds = %bitcast_concatenate_fusion.1.clone.loop_header.dim.0
  store i64 0, ptr %bitcast_concatenate_fusion.1.clone.invar_address.dim.1, align 4
  br label %bitcast_concatenate_fusion.1.clone.loop_header.dim.1

bitcast_concatenate_fusion.1.clone.loop_header.dim.1: ; preds = %concatenate.3.merge, %bitcast_concatenate_fusion.1.clone.loop_body.dim.0
  %bitcast_concatenate_fusion.1.clone.indvar.dim.1 = load i64, ptr %bitcast_concatenate_fusion.1.clone.invar_address.dim.1, align 4
  %3 = icmp uge i64 %bitcast_concatenate_fusion.1.clone.indvar.dim.1, 64
  br i1 %3, label %bitcast_concatenate_fusion.1.clone.loop_exit.dim.1, label %bitcast_concatenate_fusion.1.clone.loop_body.dim.1

bitcast_concatenate_fusion.1.clone.loop_body.dim.1: ; preds = %bitcast_concatenate_fusion.1.clone.loop_header.dim.1
  br label %concatenate.pivot.16.

concat_index_from_operand_id0:                    ; preds = %concatenate.pivot.0.
  %4 = phi i64 [ 0, %concatenate.pivot.0. ]
  %5 = sub nsw i64 %bitcast_concatenate_fusion.1.clone.indvar.dim.1, %4
  %6 = add i64 %bitcast_concatenate_fusion.1.clone.indvar.dim.0, 0
  %7 = add i64 %5, 0
  %8 = getelementptr inbounds [3 x [10000 x [64 x float]]], ptr %arg0, i64 0, i64 0, i64 %6, i64 %7
  %9 = load float, ptr %8, align 4, !invariant.load !1, !noalias !5
  br label %concatenate.3.merge

concat_index_from_operand_id1:                    ; preds = %concatenate.pivot.16.4
  %10 = phi i64 [ 16, %concatenate.pivot.16.4 ]
  %11 = sub nsw i64 %bitcast_concatenate_fusion.1.clone.indvar.dim.1, %10
  %12 = add i64 %bitcast_concatenate_fusion.1.clone.indvar.dim.0, 0
  %13 = add i64 %11, 16
  %14 = getelementptr inbounds [3 x [10000 x [64 x float]]], ptr %arg0, i64 0, i64 1, i64 %12, i64 %13
  %15 = load float, ptr %14, align 4, !invariant.load !1, !noalias !5
  br label %concatenate.3.merge

concat_index_from_operand_id2:                    ; preds = %concatenate.pivot.40.5
  %16 = phi i64 [ 40, %concatenate.pivot.40.5 ]
  %17 = sub nsw i64 %bitcast_concatenate_fusion.1.clone.indvar.dim.1, %16
  %18 = add i64 %bitcast_concatenate_fusion.1.clone.indvar.dim.0, 0
  %19 = add i64 %17, 40
  %20 = getelementptr inbounds [3 x [10000 x [64 x float]]], ptr %arg0, i64 0, i64 2, i64 %18, i64 %19
  %21 = load float, ptr %20, align 4, !invariant.load !1, !noalias !5
  br label %concatenate.3.merge

concatenate.pivot.16.:                            ; preds = %bitcast_concatenate_fusion.1.clone.loop_body.dim.1
  %22 = icmp ult i64 %bitcast_concatenate_fusion.1.clone.indvar.dim.1, 16
  br i1 %22, label %concatenate.pivot.0., label %concatenate.pivot.40.

concatenate.pivot.0.:                             ; preds = %concatenate.pivot.16.
  br label %concat_index_from_operand_id0

concatenate.pivot.40.:                            ; preds = %concatenate.pivot.16.
  %23 = icmp ult i64 %bitcast_concatenate_fusion.1.clone.indvar.dim.1, 40
  br i1 %23, label %concatenate.pivot.16.4, label %concatenate.pivot.40.5

concatenate.pivot.16.4:                           ; preds = %concatenate.pivot.40.
  br label %concat_index_from_operand_id1

concatenate.pivot.40.5:                           ; preds = %concatenate.pivot.40.
  br label %concat_index_from_operand_id2

concatenate.3.merge:                              ; preds = %concat_index_from_operand_id2, %concat_index_from_operand_id1, %concat_index_from_operand_id0
  %24 = phi float [ %9, %concat_index_from_operand_id0 ], [ %15, %concat_index_from_operand_id1 ], [ %21, %concat_index_from_operand_id2 ]
  %25 = getelementptr inbounds [10000 x [64 x float]], ptr %arg1, i64 0, i64 %bitcast_concatenate_fusion.1.clone.indvar.dim.0, i64 %bitcast_concatenate_fusion.1.clone.indvar.dim.1
  store float %24, ptr %25, align 4, !alias.scope !5
  %invar.inc3 = add nuw nsw i64 %bitcast_concatenate_fusion.1.clone.indvar.dim.1, 1
  store i64 %invar.inc3, ptr %bitcast_concatenate_fusion.1.clone.invar_address.dim.1, align 4
  br label %bitcast_concatenate_fusion.1.clone.loop_header.dim.1

bitcast_concatenate_fusion.1.clone.loop_exit.dim.1: ; preds = %bitcast_concatenate_fusion.1.clone.loop_header.dim.1
  %invar.inc = add nuw nsw i64 %bitcast_concatenate_fusion.1.clone.indvar.dim.0, 1
  store i64 %invar.inc, ptr %bitcast_concatenate_fusion.1.clone.invar_address.dim.0, align 4
  br label %bitcast_concatenate_fusion.1.clone.loop_header.dim.0, !llvm.loop !8

bitcast_concatenate_fusion.1.clone.loop_exit.dim.0: ; preds = %bitcast_concatenate_fusion.1.clone.loop_header.dim.0
  br label %return

return:                                           ; preds = %bitcast_concatenate_fusion.1.clone.loop_exit.dim.0
  ret ptr null
}

attributes #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"xla_dylib_index", i64 1}
!1 = !{}
!2 = !{i64 7680000}
!3 = !{i64 64}
!4 = !{i64 2560000}
!5 = !{!6}
!6 = !{!"result slice: {index:2, offset:0, size:2560000}", !7}
!7 = !{!"XLA host kernel bitcast_concatenate_fusion.1.clone AA domain"}
!8 = distinct !{!8, !9}
!9 = !{!"llvm.loop.unroll.disable"}
