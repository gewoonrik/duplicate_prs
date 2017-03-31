db.issue_comments.find({ "body": /.*\#([0-9]+).*/}).forEach(function(obj){
        var pr = db.pull_requests.findOne({"repo":obj.repo, "owner":obj.owner, "number":obj.issue_id});
        if(pr != null) {
                print(obj.owner+","+obj.repo+","+obj.issue_id+","+obj.id)
        }
});
