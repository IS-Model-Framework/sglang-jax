; ModuleID = '__compute_module_dynamic-slice.5.clone_elemental_kernel_module'
source_filename = "__compute_module_dynamic-slice.5.clone_elemental_kernel_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@dynamic-slice.5.clone_parallel_bounds = private unnamed_addr constant [4 x [2 x [2 x i64]]] [[2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 0, i64 2500]], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 2500, i64 5000]], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 5000, i64 7500]], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 7500, i64 10000]]]

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define noalias noundef ptr @dynamic-slice.5.clone_kernel(ptr readonly captures(none) %0) local_unnamed_addr #0 {
  %tid_gep = getelementptr inbounds nuw i8, ptr %0, i64 8
  %tids = load ptr, ptr %tid_gep, align 8
  %tid_x = load i64, ptr %tids, align 4
  %args_gep = getelementptr inbounds nuw i8, ptr %0, i64 24
  %args = load ptr, ptr %args_gep, align 8
  %arg0 = load ptr, ptr %args, align 8, !invariant.load !1, !dereferenceable !2, !align !3
  %arg3_gep = getelementptr i8, ptr %args, i64 48
  %arg3 = load ptr, ptr %arg3_gep, align 8, !invariant.load !1, !dereferenceable !4, !align !3
  %arg4_gep = getelementptr i8, ptr %args, i64 64
  %arg4 = load ptr, ptr %arg4_gep, align 8, !invariant.load !1, !dereferenceable !5, !align !3
  %lo_dim_1_gep = getelementptr inbounds [4 x [2 x [2 x i64]]], ptr @dynamic-slice.5.clone_parallel_bounds, i64 0, i64 %tid_x, i64 1, i64 0
  %up_dim_1_gep = getelementptr inbounds [4 x [2 x [2 x i64]]], ptr @dynamic-slice.5.clone_parallel_bounds, i64 0, i64 %tid_x, i64 1, i64 1
  %lo_dim_1 = load i64, ptr %lo_dim_1_gep, align 16
  %up_dim_1 = load i64, ptr %up_dim_1_gep, align 8
  %.not1116 = icmp ult i64 %lo_dim_1, %up_dim_1
  %2 = load i32, ptr %arg3, align 64
  %narrow13 = tail call i32 @llvm.smax.i32(i32 %2, i32 0)
  %3 = tail call i32 @llvm.umin.i32(i32 %narrow13, i32 48)
  %dynamic-slice.5.clone.start_idx2 = zext nneg i32 %3 to i64
  br i1 %.not1116, label %dynamic-slice.5.clone.loop_header.dim.1.preheader.us, label %return

dynamic-slice.5.clone.loop_header.dim.1.preheader.us: ; preds = %1
  %arg1_gep = getelementptr i8, ptr %args, i64 16
  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !1, !dereferenceable !4, !align !3
  %4 = load i32, ptr %arg1, align 64
  %narrow = tail call i32 @llvm.smax.i32(i32 %4, i32 0)
  %5 = tail call i32 @llvm.umin.i32(i32 %narrow, i32 2)
  %dynamic-slice.5.clone.start_idx0 = zext nneg i32 %5 to i64
  %6 = add nuw nsw i64 %dynamic-slice.5.clone.start_idx2, 8
  br label %dynamic-slice.5.clone.loop_header.dim.2.preheader.us

dynamic-slice.5.clone.loop_header.dim.2.preheader.us: ; preds = %dynamic-slice.5.clone.loop_header.dim.1.preheader.us, %dynamic-slice.5.clone.loop_header.dim.2.preheader.us
  %dynamic-slice.5.clone.invar_address.dim.1.017.us = phi i64 [ %lo_dim_1, %dynamic-slice.5.clone.loop_header.dim.1.preheader.us ], [ %invar.inc9.us, %dynamic-slice.5.clone.loop_header.dim.2.preheader.us ]
  %7 = getelementptr inbounds [3 x [10000 x [64 x float]]], ptr %arg0, i64 0, i64 %dynamic-slice.5.clone.start_idx0, i64 %dynamic-slice.5.clone.invar_address.dim.1.017.us, i64 %dynamic-slice.5.clone.start_idx2
  %8 = getelementptr inbounds [1 x [10000 x [16 x float]]], ptr %arg4, i64 0, i64 0, i64 %dynamic-slice.5.clone.invar_address.dim.1.017.us, i64 0
  %9 = load <8 x float>, ptr %7, align 4, !invariant.load !1, !noalias !6
  store <8 x float> %9, ptr %8, align 64, !alias.scope !6
  %10 = getelementptr inbounds [3 x [10000 x [64 x float]]], ptr %arg0, i64 0, i64 %dynamic-slice.5.clone.start_idx0, i64 %dynamic-slice.5.clone.invar_address.dim.1.017.us, i64 %6
  %11 = getelementptr inbounds [1 x [10000 x [16 x float]]], ptr %arg4, i64 0, i64 0, i64 %dynamic-slice.5.clone.invar_address.dim.1.017.us, i64 8
  %12 = load <8 x float>, ptr %10, align 4, !invariant.load !1, !noalias !6
  store <8 x float> %12, ptr %11, align 32, !alias.scope !6
  %invar.inc9.us = add nuw nsw i64 %dynamic-slice.5.clone.invar_address.dim.1.017.us, 1
  %exitcond.not = icmp eq i64 %invar.inc9.us, %up_dim_1
  br i1 %exitcond.not, label %return, label %dynamic-slice.5.clone.loop_header.dim.2.preheader.us, !llvm.loop !9

return:                                           ; preds = %dynamic-slice.5.clone.loop_header.dim.2.preheader.us, %1
  ret ptr null
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.umin.i32(i32, i32) #1

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"xla_dylib_index", i64 0}
!1 = !{}
!2 = !{i64 7680000}
!3 = !{i64 64}
!4 = !{i64 4}
!5 = !{i64 640000}
!6 = !{!7}
!7 = !{!"result slice: {index:1, offset:0, size:640000}", !8}
!8 = !{!"XLA host kernel dynamic-slice.5.clone_kernel AA domain"}
!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.unroll.disable"}
