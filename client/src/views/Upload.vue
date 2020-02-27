<template>
    <div  class="window-height window-width row justify-center items-center" :style="{backgroundSize: 'cover', backgroundImage: 'url(' + require('@/bg.jpg') + ')' }"
    style="background: linear-gradient(135deg,  #151515  0%,#282828 100%)">
      <div class="column">
      <div class="row">
        <q-card square dark class="q-pa-md q-ma-none no-shadow bg-grey-10" style="width:40vw;">
          <q-card-section class="q-mt-xl q-mb-md">
            <p class="text-center text-weight-bolder text-white">Please select music file and click upload,then categorisation automatically start using AI engine !!</p>
          </q-card-section>
          <q-card-section color="white">
            <q-file rounded outlined bottom-slots v-model="file" label=" Select misic file from here or drag and drop file !!" counter max-files="12" bordercolor="white" @change="selectedFile" id="file">
              <template color="white" v-slot:before >
                <q-icon name="attachment" color="white"/>
              </template>

              <template v-slot:append color="white">
                <q-icon v-if="file !== null" name="close" @click.stop="file = null" class="cursor-pointer" color="white"/>
                <q-icon name="search" @click.stop color="white" />
              </template>
              <template v-slot:hint id="form">
                Field hint
              </template>
            </q-file>
          </q-card-section>
          <q-card-actions>
            <div class="row  items-center full-width">
              <div class="col-12">
                <q-btn outline rounded size="md" color="red-4" class="text-white full-width" label="Upload" type="submit" @click="submitFile" style="width:50%"/>
              </div>
            </div>
          </q-card-actions>
          <q-card-section>
            <p class="text-center text-caption text-weight-light text-grey">Created by AMG</p>
          </q-card-section>
          <q-card-section class="row justify-center full-width">
          <img src="@/logo.png" class="example1"/>
          </q-card-section>
        </q-card>
      </div>
    </div>

    </div>
</template>
<script>

export default {

  data () {
    return {
      file : null
    }
  },
  methods : {
    selectedFile: function() {
      // 選択された File の情報を保存しておく
      this.file = this.$refs.file.files[0];
    },
    submitFile:async function(){
      let formData = new FormData();
      formData.append('file', this.file);
      let result = "nothing"
      await this.$axios.post('yoururl',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
            'Access-Control-Allow-Origin': '*',
          }
       }
      ).then(function(res){
        result = res.data;
        //this.$router.push({name: 'result', params: {res: res.data}})
      })
      .catch(function(){
        console.log('FAILURE!!');
      });
      this.$router.push({name: 'result', params: {res: result}})
    }
  }
}
</script>

<style>
  .q-field--outlined .q-field__control:before {
    border: 1px solid rgba(255,255,255);
  }
  .q-field__native{
    color: white;
  }
  .q-field__label{
    color: white;
  }
  .q-field__messages{
    color: white;
  }
  .q-field__counter{
    color: white;
  }
  img.example1 {
    width: 100px;
    height: 65px;
  }
</style>
