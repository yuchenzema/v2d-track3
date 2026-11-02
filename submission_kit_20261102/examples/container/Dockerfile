# V2D Challenge submission container. See CONTAINER_SUBMISSION.md.
#
# Build from a directory holding submission/ laid out as /submission will be:
#
#   submission/
#     submission.json   what each leaderboard submission was made from (schema in the guide)
#     run.sh            entry point (start from examples/container/run.sh)
#     code/             your repository, checked out at the submitted commit, .git included
#     kit/              the submission kit copy that packed and uploaded the submissions
#     verifier_kit/     the kit version that ships CONTAINER_SUBMISSION.md (for run.sh verify)
#     submitted/        the exact files uploaded to Kaggle, and their upload receipts
#     weights/          Tracks 2 and 3: one policy checkpoint per scored episode
#     outputs/          evaluator rollouts (Tracks 2 and 3) or final reconstructions (Track 1)
#
#   docker build -t TEAM-v2d-submission:FINAL -f Dockerfile .
#
# Start from the image you evaluated (Tracks 2 and 3) or reconstructed (Track 1) in, so the
# runtime that produced the leaderboard files is the one in the container.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# The verifier and entry point need git and the kit's Python requirements in the image.
# Keep file modes as committed: git would report a changed mode as a modified file.
# Make everything readable for reviewers who run the container as a non-root user. a+rX adds
# read access, and execute only on directories and already-executable files, so git still sees
# a clean checkout.
COPY submission /submission
RUN chmod 755 /submission/run.sh && chmod -R a+rX /submission
WORKDIR /submission
CMD ["/submission/run.sh", "verify"]
