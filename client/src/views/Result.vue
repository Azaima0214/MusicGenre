<template>
    <div  class="window-height window-width row justify-center items-center"
    style="background: linear-gradient(135deg,  #151515  0%,#282828 100%)">

      <div class="column">
      <div class="row">
        <q-card square dark class="q-pa-md q-ma-none no-shadow bg-grey-10" style="width:40vw;">
          <q-card-section class="q-mt-xl q-mb-md">
            <p class="text-center text-weight-bolder text-white">Result Your Music Genre</p>
          </q-card-section>
          <q-card-section color="white">
            <h3 class="text-center text-weight-bolder text-white">{{ $route.params.res }}</h3>
          </q-card-section>
          <q-card-actions>
            <div class="row  items-center full-width">
              <div class="col-12">
                <q-btn outline rounded size="md" color="red-4" class="text-white full-width" label="BACK PAGE" type="submit" @click="BackPage" style="width:50%"/>
              </div>
            </div>
          </q-card-actions>
          <q-card-section>
            <p class="text-center text-caption text-weight-light text-grey">Created by Test_Name</p>
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
      result : 'test'
    }
  },
  methods : {
    BackPage: function() {
      // 選択された File の情報を保存しておく
      this.$router.push({name: 'Upload'});
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
</style>