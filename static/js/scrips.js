
$(document).ready(function(){
    $('#videoContainer').hide();
    function stopVideo() {
        $('#videoContainer').hide();
        $('#start_btn').removeAttr('disabled');
        $('#stop_btn').attr("disabled",true);
      }
    
      function startVideo() {
        $('#videoContainer').show();
        $('#start_btn').attr("disabled",true)
        $('#stop_btn').removeAttr('disabled');
      }

      $("#start_btn").on('click',startVideo);
      $("#stop_btn").on('click',stopVideo);
 });
